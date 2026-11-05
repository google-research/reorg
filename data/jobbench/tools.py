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
"""JobBench tools (App. A.4, Table "Benchmark-specific worker and submission settings").

    code_exec             sandboxed shell; direct network commands are refused. A stakeholder
                          or Loop-In phase that must write one output file gets 50 calls before
                          the file exists and 4 after it; worker actions have no call ceiling.
    finish                every final deliverable path; an action with an output file must
                          submit exactly that path
    abandon_task_finish   a genuine hard blocker
    web_search            Gemini 3.5 Flash with Google Search grounding, up to five results
    fetch_web_page        readable text of one public page
    download_file         one public file into evidence/

The three retrieval tools are the worker's on main tasks only, under a per-action budget.
They refuse queries and URLs for JobBench itself (its name, task IDs, task cards, rubrics,
dataset hosts, its arXiv ID), quoted task text, and private or local hosts. Search
calls the Gemini Developer API (GEMINI_API_KEY), also for vLLM workers. Tool names,
descriptions, parameters and refusal texts are model-facing.
"""

from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import re
import time
from html import escape
from html.parser import HTMLParser
from pathlib import Path
from typing import Annotated
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, Field
from stirrup import Tool, ToolResult, ToolUseCountMetadata
from stirrup.core.agent import _SESSION_STATE
from stirrup.tools.code_backends.base import CommandResult
from stirrup.utils.text import truncate_msg

from reorg.llm import append_jsonl, gemini_generate
from reorg.worker import WorkspaceCodeExec

MAX_TOOL_TEXT = 30000
BLOCKED_HOST_SUFFIXES = ("huggingface.co", "hf.co", "hf-mirror.com")
BLOCKED_URL_MARKERS = ("jobbench", "job-bench", "task_card.md", "rubrics.json", "2605.26329")
BLOCKED_QUERY_MARKERS = ("jobbench", "job bench benchmark", "job-bench", "task_card.md", "rubrics.json",
                         "site:huggingface.co", "site:hf.co", "2605.26329")
DIRECT_NETWORK_PATTERNS = (
    re.compile(r"^\s*(sudo\s+)?(curl|wget|aria2c|http|https|lynx)\b", re.I),
    re.compile(r"^\s*git\s+clone\b", re.I),
    re.compile(r"^\s*((python\S*\s+-m\s+)?pip|conda|mamba|apt|apt-get)\s+install\b", re.I),
    re.compile(r"^\s*(bash\s+-c\s+)?[^\n]*/dev/(tcp|udp)/", re.I),
)
QUOTED_SEGMENT_RE = re.compile(r'"([^"]{12,})"|“([^”]{12,})”|\'([^\']{12,})\'|‘([^’]{12,})’')
OUTPUT_ROOT = "current_artifact"                # where the worker's final files live
SKIPPED_OUTPUT_PREFIXES = ("test", "tmp", "sample", "dummy", ".")


# ---- guards --------------------------------------------------------------------

def _normalize(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


def query_block_reason(query: str, *, task_id: str, task_text: str) -> str | None:
    lowered = query.lower()
    if task_id.lower() in lowered:
        return "JobBench task IDs cannot be used in web search"
    if any(marker in lowered for marker in BLOCKED_QUERY_MARKERS):
        return "Benchmark pages, task cards, rubrics, and public task copies are blocked"
    normalized_task = _normalize(task_text)
    for match in QUOTED_SEGMENT_RE.finditer(query):
        segment = _normalize(next(group for group in match.groups() if group))
        if segment and segment in normalized_task:
            return "Quoted task-prompt searches are blocked"
    return None


def url_block_reason(url: str, *, task_id: str) -> str | None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return "Only public HTTP(S) URLs are allowed"
    host = (parsed.hostname or "").lower().rstrip(".")
    if not host:
        return "URL has no hostname"
    if host == "localhost" or host.endswith(".localhost"):
        return "Local network URLs are blocked"
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None
    if address and (address.is_private or address.is_loopback or address.is_link_local or address.is_reserved):
        return "Private and local network URLs are blocked"
    if any(host == suffix or host.endswith(f".{suffix}") for suffix in BLOCKED_HOST_SUFFIXES):
        return "Public JobBench dataset hosts are blocked"
    lowered = url.lower()
    if task_id.lower() in lowered or any(marker in lowered for marker in BLOCKED_URL_MARKERS):
        return "Benchmark pages, task cards, rubrics, and public task copies are blocked"
    return None


class RetrievalBudget:
    """Counts of one action's retrieval calls; a call past its limit is refused, not made."""

    def __init__(self, *, search: int, fetch: int, download: int, byte_limit: int, log_path: Path) -> None:
        self.limits = {"search": search, "fetch": fetch, "download": download}
        self.counts = dict.fromkeys(self.limits, 0)
        self.byte_limit = byte_limit
        self.downloaded_bytes = 0
        self.log_path = log_path

    def acquire(self, kind: str) -> str | None:
        self.counts[kind] += 1
        if self.counts[kind] > self.limits[kind]:
            return f"{kind} limit reached ({self.limits[kind]} per task)"
        return None

    def add_bytes(self, count: int) -> str | None:
        if self.downloaded_bytes + count > self.byte_limit:
            return f"download byte limit would be exceeded ({self.byte_limit} bytes per task)"
        self.downloaded_bytes += count
        return None

    def log(self, row: dict) -> None:
        append_jsonl(self.log_path, {"timestamp": time.time(), **row})


# ---- code execution ------------------------------------------------------------

class GuardedCodeExec(WorkspaceCodeExec):
    """code_exec in the action sandbox with call guards.

    `outputs` are the files a stakeholder or Loop-In phase must write. Until they exist it
    gets `pre_output_limit` calls (then only calls that write them); once they exist,
    `post_output_limit` more. The worker (outputs=None, post_output_limit=None) has no
    ceiling. Every result after outputs exist carries a reminder to finish.
    """

    def __init__(self, sandbox: Path, *, log_path: Path, outputs: list[str] | None, pre_output_limit: int,
                 post_output_limit: int | None, shell_timeout: int) -> None:
        super().__init__(sandbox, shell_timeout)
        self.log_path = log_path
        self.outputs = outputs
        self.pre_output_limit = pre_output_limit
        self.post_output_limit = post_output_limit
        self.calls = 0
        self.post_output_calls = 0

    def _has_outputs(self) -> bool:
        if self.outputs is not None:
            return all((self._temp_dir / path).is_file() and (self._temp_dir / path).stat().st_size > 0
                       for path in self.outputs)
        root = self._temp_dir / OUTPUT_ROOT
        return root.is_dir() and any(path.is_file() and path.stat().st_size > 0
                                     and not path.name.lower().startswith(SKIPPED_OUTPUT_PREFIXES)
                                     for path in root.rglob("*"))

    def _refuse(self, cmd: str, status: str, stderr: str, error_kind: str, **row) -> CommandResult:
        append_jsonl(self.log_path, {"timestamp": time.time(), "status": status, **row, "command": cmd})
        return CommandResult(exit_code=1, stdout="", stderr=stderr, error_kind=error_kind)

    async def run_command(self, cmd: str, *, timeout: int | None = None) -> CommandResult:
        self.calls += 1
        if (self.outputs and not self._has_outputs() and self.calls > self.pre_output_limit
                and not any(path in cmd for path in self.outputs)):
            return self._refuse(cmd, "blocked_pending_finalization",
                                f"The analysis call budget is exhausted. Create {self.outputs} now in one "
                                "code_exec call, then call finish.", "finish_required")
        if self._has_outputs():
            self.post_output_calls += 1
            if self.post_output_limit is not None and self.post_output_calls > self.post_output_limit:
                return self._refuse(cmd, "blocked_after_outputs",
                                    f"Final artifacts already exist under {OUTPUT_ROOT}/ and the post-output tool "
                                    "budget is exhausted. Call finish now with every final deliverable path.",
                                    "finish_required")
        blocked = next((pattern.pattern for pattern in DIRECT_NETWORK_PATTERNS if pattern.search(cmd)), None)
        if blocked:
            return self._refuse(cmd, "blocked_network",
                                "Direct network retrieval is disabled. Use web_search, fetch_web_page, or download_file.",
                                "direct_network_blocked", pattern=blocked)
        started = time.time()
        result = await super().run_command(cmd, timeout=timeout)
        reminder = ""
        if self._has_outputs():
            if self.post_output_limit is None:
                reminder = (f"\n<jobbench_finalization_reminder>Final artifacts exist under {OUTPUT_ROOT}/. "
                            "Continue only with concrete validation or fixes, then call finish as soon as the "
                            "complete submission passes. No separate post-output tool-call ceiling is active for "
                            "this owner action.</jobbench_finalization_reminder>")
            else:
                remaining = max(0, self.post_output_limit - self.post_output_calls)
                reminder = (f"\n<jobbench_finalization_reminder>Final artifacts exist under {OUTPUT_ROOT}/. "
                            f"You have at most {remaining} more code_exec calls for bounded validation or concrete "
                            "fixes. Call finish as soon as they pass.</jobbench_finalization_reminder>")
        elif self.outputs and self.calls >= max(1, self.pre_output_limit - 2):
            reminder = ("\n<jobbench_finalization_reminder>Complete the review now: write "
                        f"{self.outputs} and call finish.</jobbench_finalization_reminder>")
        append_jsonl(self.log_path, {"timestamp": started, "status": "executed",
                                     "duration_seconds": round(time.time() - started, 3), "exit_code": result.exit_code,
                                     "error_kind": result.error_kind, "command": cmd})
        if reminder:
            return CommandResult(exit_code=result.exit_code, stdout=result.stdout + reminder, stderr=result.stderr,
                                 error_kind=result.error_kind, advice=result.advice)
        return result


# ---- finish tools --------------------------------------------------------------
# Parameter class names are logged as AgentRun.finish_type; runtime.py keys on the "Abandon" prefix.

class JobBenchFinishParams(BaseModel):
    reason: Annotated[str, Field(description="Brief summary of the completed work.")]
    paths: Annotated[list[str], Field(description="Relative paths to every final deliverable file.")]


class AbandonTaskFinishParams(BaseModel):
    reason: Annotated[str, Field(description="Brief reason the task cannot be completed.")]


def finish_tool(required: list[str] | None) -> Tool[JobBenchFinishParams, ToolUseCountMetadata]:
    """finish(reason, paths): existing relative files with unique basenames; exactly `required` if given."""

    async def executor(params: JobBenchFinishParams) -> ToolResult[ToolUseCountMetadata]:
        if not params.paths:
            return ToolResult(content="ERROR: submit at least one final deliverable file", success=False)
        exec_env = _SESSION_STATE.get().exec_env
        errors = []
        if required is not None and sorted(params.paths) != sorted(required):
            errors.append(f"submit exactly these paths: {required}")
        basenames: set[str] = set()
        for text in params.paths:
            path = Path(text)
            if path.is_absolute() or ".." in path.parts or not path.parts:
                errors.append(f"path must stay inside the workspace: {text}")
                continue
            if path.name in basenames:
                errors.append(f"duplicate deliverable basename: {path.name}")
            basenames.add(path.name)
            if await exec_env.is_directory(text):
                errors.append(f"path is a directory: {text}")
            elif not await exec_env.file_exists(text):
                errors.append(f"file does not exist: {text}")
        if errors:
            return ToolResult(content="ERROR: " + "; ".join(errors), success=False)
        return ToolResult(content=params.reason, success=True, metadata=ToolUseCountMetadata())

    return Tool(name="finish",
                description="Submit every completed JobBench deliverable. Paths must be relative files, not directories.",
                parameters=JobBenchFinishParams, executor=executor)


async def _abandon(params: AbandonTaskFinishParams) -> ToolResult[ToolUseCountMetadata]:
    return ToolResult(content=params.reason, success=True, metadata=ToolUseCountMetadata())


ABANDON_TOOL = Tool(name="abandon_task_finish", description="Use only when the task genuinely cannot be completed.",
                    parameters=AbandonTaskFinishParams, executor=_abandon)


# ---- retrieval -----------------------------------------------------------------

class SearchParams(BaseModel):
    query: Annotated[str, Field(description="A focused public-web search query.")]


class URLParams(BaseModel):
    url: Annotated[str, Field(description="Public HTTP(S) URL returned by web_search or named in the task.")]


class DownloadParams(URLParams):
    filename: Annotated[str, Field(description="Safe filename to save under evidence/.")]


class RetrievalMetadata(BaseModel):
    num_uses: int = 1
    bytes_received: int = 0
    urls: list[str] = Field(default_factory=list)

    def __add__(self, other: "RetrievalMetadata") -> "RetrievalMetadata":
        return RetrievalMetadata(num_uses=self.num_uses + other.num_uses,
                                 bytes_received=self.bytes_received + other.bytes_received,
                                 urls=self.urls + other.urls)


def render_search(data: dict, *, task_id: str) -> tuple[str, list[str]]:
    """Up to five grounding sources as <result> XML, dropping blocked URLs and benchmark pages."""
    candidates = data.get("candidates") or []
    metadata = (candidates[0].get("groundingMetadata") if candidates else {}) or {}
    parts = ((candidates[0].get("content") or {}).get("parts") if candidates else []) or []
    answer = "".join(str(part.get("text") or "") for part in parts if "text" in part and not part.get("thought"))
    descriptions: dict[int, str] = {}
    for support in metadata.get("groundingSupports") or []:
        text = str((support.get("segment") or {}).get("text") or "")
        for index in support.get("groundingChunkIndices") or []:
            descriptions.setdefault(int(index), text)
    rows, urls = [], []
    for index, chunk in enumerate(metadata.get("groundingChunks") or []):
        web = chunk.get("web") or {}
        title, url = str(web.get("title") or ""), str(web.get("uri") or "")
        description = descriptions.get(index) or answer[:600]
        if url_block_reason(url, task_id=task_id):
            continue
        if any(marker in f"{title} {description}".lower() for marker in BLOCKED_QUERY_MARKERS):
            continue
        rows.append(f"<result>\n<title>{escape(title)}</title>\n<url>{escape(url)}</url>\n"
                    f"<description>{escape(description)}</description>\n</result>")
        urls.append(url)
        if len(rows) == 5:
            break
    return truncate_msg("<results>\n" + "\n".join(rows) + "\n</results>", MAX_TOOL_TEXT), urls


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.hidden = 0

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in {"script", "style", "noscript"}:
            self.hidden += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self.hidden:
            self.hidden -= 1

    def handle_data(self, data: str) -> None:
        if not self.hidden and data.strip():
            self.parts.append(data.strip())


async def _guarded_get(url: str, *, task_id: str, timeout: float, max_bytes: int) -> httpx.Response:
    """GET with the URL guard applied to the request and every redirect hop, and a size cap."""
    if reason := url_block_reason(url, task_id=task_id):
        raise ValueError(reason)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        async with client.stream("GET", url, headers={"User-Agent": "JobBench-Stirrup/1.0"}) as streamed:
            for hop in [*streamed.history, streamed]:
                if reason := url_block_reason(str(hop.url), task_id=task_id):
                    raise ValueError(f"Blocked redirect target: {reason}")
            streamed.raise_for_status()
            length = streamed.headers.get("content-length")
            if length is not None and int(length) > max_bytes:
                raise ValueError(f"Response exceeded {max_bytes} bytes")
            chunks, received = [], 0
            async for chunk in streamed.aiter_bytes():
                received += len(chunk)
                if received > max_bytes:
                    raise ValueError(f"Response exceeded {max_bytes} bytes")
                chunks.append(chunk)
            headers = [(name, value) for name, value in streamed.headers.multi_items()
                       if name.lower() not in {"content-encoding", "content-length", "transfer-encoding"}]
            return httpx.Response(streamed.status_code, headers=headers, content=b"".join(chunks),
                                  request=streamed.request, extensions=streamed.extensions, history=streamed.history)


def retrieval_tools(*, task_id: str, task_text: str, budget: RetrievalBudget, settings: dict) -> list[Tool]:
    """[web_search, fetch_web_page, download_file] for one worker action on a main task."""

    async def search(params: SearchParams) -> ToolResult[RetrievalMetadata]:
        reason = budget.acquire("search") or query_block_reason(params.query, task_id=task_id, task_text=task_text)
        if reason:
            budget.log({"kind": "search_blocked", "query": params.query, "reason": reason})
            return ToolResult(content=f"<results><error>{escape(reason)}</error></results>", success=False,
                              metadata=RetrievalMetadata())
        payload = {"contents": [{"role": "user", "parts": [{"text": "Search the public web for authoritative "
                                                                    "sources.\n\nQuery: " + params.query}]}],
                   "tools": [{"google_search": {}}],
                   "generationConfig": {"maxOutputTokens": 2048, "temperature": 0.0,
                                        "thinkingConfig": {"thinkingLevel": settings["search_thinking_level"]}}}
        started = time.time()
        try:
            data = await asyncio.to_thread(gemini_generate, settings["search_model"], payload,
                                           timeout=settings["search_timeout_seconds"], log_path=budget.log_path,
                                           kind="web_search")
        except Exception as exc:
            budget.log({"kind": "search_error", "query": params.query, "error": repr(exc)})
            return ToolResult(content=f"<results><error>{escape(repr(exc))}</error></results>", success=False,
                              metadata=RetrievalMetadata())
        xml, urls = render_search(data, task_id=task_id)
        budget.log({"kind": "search", "query": params.query, "duration_seconds": round(time.time() - started, 3),
                    "urls": urls})
        return ToolResult(content=xml, success=bool(urls), metadata=RetrievalMetadata(urls=urls))

    async def fetch(params: URLParams) -> ToolResult[RetrievalMetadata]:
        if error := budget.acquire("fetch"):
            return ToolResult(content=f"<webpage><error>{escape(error)}</error></webpage>", success=False,
                              metadata=RetrievalMetadata())
        started = time.time()
        try:
            response = await _guarded_get(params.url, task_id=task_id, timeout=settings["fetch_timeout_seconds"],
                                          max_bytes=5_000_000)
            content_type = response.headers.get("content-type", "").lower()
            if "html" in content_type:
                parser = _TextExtractor()
                parser.feed(response.text)
                text = "\n".join(parser.parts)
            elif content_type.startswith("text/") or "json" in content_type or "xml" in content_type:
                text = response.text
            else:
                raise ValueError(f"Binary content type {content_type!r}; use download_file")
            final_url = str(response.url)
            budget.log({"kind": "fetch", "requested_url": params.url, "final_url": final_url,
                        "status_code": response.status_code, "bytes": len(response.content),
                        "duration_seconds": round(time.time() - started, 3)})
            return ToolResult(content=f"<webpage><url>{escape(final_url)}</url>\n"
                                      f"{escape(truncate_msg(text, MAX_TOOL_TEXT))}</webpage>",
                              success=True, metadata=RetrievalMetadata(bytes_received=len(response.content),
                                                                       urls=[final_url]))
        except Exception as exc:
            budget.log({"kind": "fetch_error", "requested_url": params.url, "error": repr(exc)})
            return ToolResult(content=f"<webpage><error>{escape(repr(exc))}</error></webpage>", success=False,
                              metadata=RetrievalMetadata())

    async def download(params: DownloadParams) -> ToolResult[RetrievalMetadata]:
        error = budget.acquire("download")
        filename = Path(params.filename).name
        if error or filename != params.filename or filename in {"", ".", ".."}:
            reason = error or "filename must be a simple basename"
            return ToolResult(content=f"<download><error>{escape(reason)}</error></download>", success=False,
                              metadata=RetrievalMetadata())
        started = time.time()
        try:
            response = await _guarded_get(params.url, task_id=task_id, timeout=settings["fetch_timeout_seconds"],
                                          max_bytes=settings["max_download_file_bytes"])
            size = len(response.content)
            if byte_error := budget.add_bytes(size):
                raise ValueError(byte_error)
            destination = f"evidence/{filename}"
            await _SESSION_STATE.get().exec_env.write_file_bytes(destination, response.content)
            digest = hashlib.sha256(response.content).hexdigest()
            final_url = str(response.url)
            budget.log({"kind": "download", "requested_url": params.url, "final_url": final_url,
                        "path": destination, "bytes": size, "sha256": digest,
                        "content_type": response.headers.get("content-type"),
                        "duration_seconds": round(time.time() - started, 3)})
            return ToolResult(content=(f"<download><path>{destination}</path><bytes>{size}</bytes>"
                                       f"<sha256>{digest}</sha256><url>{escape(final_url)}</url></download>"),
                              success=True, metadata=RetrievalMetadata(bytes_received=size, urls=[final_url]))
        except Exception as exc:
            budget.log({"kind": "download_error", "requested_url": params.url, "filename": filename,
                        "error": repr(exc)})
            return ToolResult(content=f"<download><error>{escape(repr(exc))}</error></download>", success=False,
                              metadata=RetrievalMetadata())

    return [
        Tool(name="web_search",
             description=("Search the public web for authoritative real-world sources. Public JobBench pages, task "
                          "copies, rubrics, task cards, and benchmark dataset hosts are blocked."),
             parameters=SearchParams, executor=search),
        Tool(name="fetch_web_page",
             description="Fetch readable text from an allowed public web page. Use download_file for PDFs or datasets.",
             parameters=URLParams, executor=fetch),
        Tool(name="download_file",
             description="Download an allowed public source file into evidence/. Provide a simple filename, not a path.",
             parameters=DownloadParams, executor=download),
    ]
