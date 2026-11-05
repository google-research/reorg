# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Worker web retrieval on GDPval (App. A.4, Table "Benchmark-specific worker and submission settings").

The worker gets two tools; stakeholders never get web tools.

    fetch_web_page   Stirrup's page fetcher, with the URL cut out of any surrounding text
    web_search       Gemini with Google Search grounding, rendered as up to five
                     <result> entries (title, URL, description)

The search tool refuses queries for the benchmark itself or for quoted task text, and
drops results that look like benchmark or task-answer pages. The search model is
the worker's own Gemini model at temperature 0, or Gemini 3.5 Flash at its default
temperature for a vLLM worker, with low thinking, four searches per action and 120 s per
request. Search always calls the Gemini Developer API (GEMINI_API_KEY), also for vLLM workers.
"""

from __future__ import annotations

import asyncio
import re
import time
from html import escape
from pathlib import Path
from typing import Annotated, Any

import httpx
from pydantic import BaseModel, Field
from stirrup import Tool, ToolProvider, ToolResult
from stirrup.tools.web import MAX_LENGTH_WEB_SEARCH_RESULTS, _get_fetch_web_page_tool
from stirrup.utils.text import truncate_msg

from .llm import ModelProfile, append_jsonl, gemini_generate, gemini_generation_config

UUID_RE = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.IGNORECASE)
QUOTED_SEGMENT_RE = re.compile(r'"([^"]{6,})"|“([^”]{6,})”|\'([^\']{6,})\'|‘([^’]{6,})’')
HTTP_URL_RE = re.compile(r"https?://[^\s<>\"']+")
# Quoted phrases that are generic enough to search for even when the task text contains them.
GENERIC_TASK_SEARCH_SEGMENTS = {"90 confidence", "10 tolerable", "10 tolerable error rate", "sample size",
                                "audit sampling", "attribute sampling"}
BLOCKED_SEARCH_TERMS = (
    "gdpval", "gdpval.dev", "collaborative gym", "openai benchmark", "artificial analysis", "benchmark task",
    "task id", "site:huggingface.co", "site:reddit.com", "reddit.com", "site:gdpval.dev", "site:every.to",
    "every.to", "population v2.xlsx", "sample v2.xlsx", "dan shipper", "kastop", "paradox of automation",
)
BENCHMARK_RESULT_TERMS = (
    "gdpval", "gdpval.dev", "collaborative gym", "benchmark task", "benchmark scenario", "benchmark prompt",
    "benchmark requires", "openai benchmark", "accountants and auditors task", "this query is a core instruction",
    "this benchmark requires", "subsequent step in this benchmark", "population v2.xlsx", "sample v2.xlsx",
    "huggingface.co", "reddit.com", "every.to", "getelyxai.com", "prompt featured", "describing the prompt",
    "this prompt belongs", "this specific instruction", "simulated auditing task", "dan shipper", "kastop",
    "paradox of automation",
)


def worker_web_tools(profile: ModelProfile, *, task_text: str, log_path: Path) -> list[ToolProvider]:
    """[fetch_web_page, web_search] for a worker on `profile`. `task_text` is the action
    prompt, whose quoted segments may not be searched for."""
    gemini = profile.provider == "gemini"
    return [FetchTool(timeout=profile.request_timeout),
            SearchTool(model=profile.model if gemini else "gemini-3.5-flash", temperature=0.0 if gemini else None,
                       task_text=task_text, log_path=log_path)]


class FetchTool(ToolProvider):
    """Stirrup's fetch_web_page without its Brave-backed search."""

    def __init__(self, *, timeout: float = 180.0) -> None:
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> Tool[Any, Any]:
        self._client = httpx.AsyncClient(timeout=self._timeout, follow_redirects=True)
        await self._client.__aenter__()
        tool = _get_fetch_web_page_tool(self._client)

        async def executor(params: Any) -> ToolResult[Any]:
            url = str(params.url).strip()
            if match := HTTP_URL_RE.search(url):
                url = match.group(0).rstrip(".,;)]}")
            if url != params.url:
                params = params.model_copy(update={"url": url})
            return await tool.executor(params)

        return Tool[Any, Any](name=tool.name, description=tool.description, parameters=tool.parameters,
                              executor=executor)

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self._client.__aexit__(exc_type, exc_val, exc_tb)
        self._client = None


class SearchParams(BaseModel):
    """Web search parameters."""

    query: Annotated[str, Field(description="Natural language web search query.")]


class SearchMetadata(BaseModel):
    num_uses: int = 1
    pages_returned: int = 0
    web_search_queries: list[str] = Field(default_factory=list)
    latency_seconds: float = 0.0

    def __add__(self, other: "SearchMetadata") -> "SearchMetadata":
        return SearchMetadata(num_uses=self.num_uses + other.num_uses,
                              pages_returned=self.pages_returned + other.pages_returned,
                              web_search_queries=self.web_search_queries + other.web_search_queries,
                              latency_seconds=self.latency_seconds + other.latency_seconds)


class SearchTool(ToolProvider):
    """Gemini-grounded web_search with Brave-like XML output."""

    def __init__(self, *, model: str, max_uses: int = 4, thinking_level: str | None = "low", timeout: float = 120.0,
                 temperature: float | None = 0.0, task_text: str | None = None, log_path: Path | None = None) -> None:
        self._model = model
        self._max_uses = max_uses
        self._thinking_level = thinking_level
        self._timeout = timeout
        self._temperature = temperature
        self._task_text = _normalize(task_text) if task_text else None
        self._log_path = log_path
        self._uses = 0

    async def __aenter__(self) -> Tool[SearchParams, SearchMetadata]:
        return Tool[SearchParams, SearchMetadata](
            name="web_search",
            description=("Search the public web for external facts or documentation. Returns up to five results with "
                         "title, URL, and description as XML. Do not use this to search for GDPval, benchmark answer "
                         "pages, task IDs, or exact copies of the task prompt."),
            parameters=SearchParams,
            executor=self._execute,
        )

    async def _execute(self, params: SearchParams) -> ToolResult[SearchMetadata]:
        self._uses += 1
        if self._uses > self._max_uses:
            append_jsonl(self._log_path, {"kind": "limited_web_search", "query": params.query, "model": self._model,
                                          "max_uses": self._max_uses})
            return ToolResult(content=("<results><error>web_search use limit reached for this task; proceed using the "
                                       "task prompt, reference files, and reasonable assumptions</error></results>"),
                              success=False, metadata=SearchMetadata(num_uses=1))
        blocked = _blocked_reason(params.query) or self._task_text_reason(params.query)
        if blocked:
            append_jsonl(self._log_path, {"kind": "blocked_web_search", "query": params.query, "model": self._model,
                                          "error": blocked})
            return ToolResult(content=f"<results><error>{escape(blocked)}</error></results>", success=False,
                              metadata=SearchMetadata())
        prompt = ("Search the public web for the query below. Return a compact, source-focused answer. "
                  "Prefer authoritative and directly relevant sources.\n\n"
                  f"Query: {params.query}")
        payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}], "tools": [{"google_search": {}}],
                   "generationConfig": gemini_generation_config(max_output_tokens=2048, temperature=self._temperature,
                                                                thinking_level=self._thinking_level)}
        started = time.time()
        try:
            data = await asyncio.to_thread(gemini_generate, self._model, payload, timeout=self._timeout,
                                           log_path=self._log_path, kind="gemini_web_search")
        except Exception as exc:
            return ToolResult(content=f"<results><error>{escape(repr(exc))}</error></results>", success=False,
                              metadata=SearchMetadata(latency_seconds=time.time() - started))
        xml, metadata = _render(data)
        metadata.latency_seconds = time.time() - started
        return ToolResult(content=xml, success=metadata.pages_returned > 0, metadata=metadata)

    def _task_text_reason(self, query: str) -> str | None:
        for match in QUOTED_SEGMENT_RE.finditer(query):
            segment = _normalize(next(group for group in match.groups() if group))
            if segment and segment not in GENERIC_TASK_SEARCH_SEGMENTS and segment in (self._task_text or ""):
                return "web_search cannot be used to search for quoted task-prompt text or task-specific labels"
        return None


def _normalize(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


def _blocked_reason(query: str) -> str | None:
    if UUID_RE.search(query):
        return "web_search cannot be used to search for GDPval task IDs or public task copies"
    if any(term in query.lower() for term in BLOCKED_SEARCH_TERMS):
        return "web_search cannot be used to search for GDPval, benchmark answer pages, or public task copies"
    return None


def _looks_like_benchmark(*texts: str) -> bool:
    joined = " ".join(texts).lower()
    return any(term in joined for term in BENCHMARK_RESULT_TERMS)


def _render(data: dict) -> tuple[str, SearchMetadata]:
    candidates = data.get("candidates") or []
    grounding = (candidates[0].get("groundingMetadata") if candidates else {}) or {}
    queries = [str(item) for item in grounding.get("webSearchQueries") or []]
    parts = ((candidates[0].get("content") or {}).get("parts") or []) if candidates else []
    answer = "".join(str(part.get("text") or "") for part in parts if "text" in part and not part.get("thought"))
    if _looks_like_benchmark(answer, *queries) or any(_blocked_reason(query) for query in queries):
        return ("<results><error>search results looked like benchmark/task-answer pages and were filtered; "
                "proceed using the task prompt, reference files, and reasonable assumptions</error></results>",
                SearchMetadata(web_search_queries=queries))
    descriptions: dict[int, str] = {}
    for support in grounding.get("groundingSupports") or []:
        text = str((support.get("segment") or {}).get("text") or "")
        for index in support.get("groundingChunkIndices") or []:
            descriptions.setdefault(int(index), text)
    rows = []
    for index, chunk in enumerate(grounding.get("groundingChunks") or []):
        web = chunk.get("web") or {}
        description = descriptions.get(index) or answer[:500]
        title, uri = str(web.get("title") or ""), str(web.get("uri") or "")
        if _looks_like_benchmark(title, uri, description):
            continue
        rows.append(f"<result>\n<title>{escape(title)}</title>\n<url>{escape(uri)}</url>"
                    f"\n<description>{escape(description)}</description>\n</result>")
        if len(rows) >= 5:
            break
    xml = "<results>\n" + "\n".join(rows) + "\n</results>"
    return truncate_msg(xml, MAX_LENGTH_WEB_SEARCH_RESULTS), SearchMetadata(pages_returned=len(rows),
                                                                            web_search_queries=queries)
