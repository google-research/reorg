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
"""Stirrup clients for the three GDPval judges (App. A.5.2).

    gemini         reorg.llm, Gemini Developer API (GEMINI_API_KEY)          Gemini 3.1 Pro
    vllm           reorg.llm, native vLLM server (VLLM_BASE_URL)              Qwen 3.6 27B
    vertex_claude  Anthropic Messages on Vertex AI (VERTEX_PROJECT, ADC)     Claude Sonnet 4.6

The Gemini and vLLM members use reorg.llm's agent clients. Only the Sonnet
member's Vertex route lives here; it takes its tool declarations and text
flattening from reorg.llm, so all three judges see the same tool schemas.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import httpx
from stirrup.core.exceptions import ContextOverflowError
from stirrup.core.models import (AssistantMessage, ChatMessage, Content, ImageContentBlock, LLMClient, Reasoning,
                                 SystemMessage, TokenUsage, Tool, ToolCall, ToolMessage, UserMessage)

from reorg import llm

VERTEX_SCOPE = "https://www.googleapis.com/auth/cloud-platform"
RETRYABLE_STATUS = {401, 408, 429, 500, 502, 503, 504}
MAX_ATTEMPTS = 5
ANTHROPIC_MAX_IMAGE_BASE64_BYTES = 5 * 1024 * 1024
ANTHROPIC_IMAGE_PIXEL_CAPS = (1_000_000, 750_000, 500_000, 250_000, 125_000)


class VertexToken:
    """Application Default Credentials, cached; `gcloud auth print-access-token` as the fallback."""

    def __init__(self) -> None:
        self._token: str | None = None
        self._expires_at = 0.0
        self._credentials: Any = None
        self._lock = threading.Lock()

    def get(self) -> str:
        with self._lock:
            if self._token and time.time() < self._expires_at - 300:
                return self._token
            try:
                import google.auth
                from google.auth.transport.requests import Request

                if self._credentials is None:
                    self._credentials, _ = google.auth.default(scopes=[VERTEX_SCOPE])
                if not self._credentials.valid:
                    self._credentials.refresh(Request())
                self._token = str(self._credentials.token)
                expiry = getattr(self._credentials, "expiry", None)
                self._expires_at = expiry.timestamp() if expiry else time.time() + 3000
            except Exception:
                result = subprocess.run(["gcloud", "auth", "print-access-token"], check=True, text=True,
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                self._token = result.stdout.strip()
                self._expires_at = time.time() + 3000
            return self._token

    def invalidate(self) -> None:
        with self._lock:
            self._token, self._expires_at, self._credentials = None, 0.0, None


def _safe_json_loads(value: str) -> Any:
    try:
        return json.loads(value) if value else {}
    except json.JSONDecodeError:
        return {}


def _anthropic_image(block: ImageContentBlock) -> dict[str, str]:
    # Shrink until the base64 payload fits Anthropic's 5 MB image limit.
    for max_pixels in ANTHROPIC_IMAGE_PIXEL_CAPS:
        header, data = block.to_base64_url(max_pixels=max_pixels).split(",", 1)
        media_type = header.removeprefix("data:").split(";", 1)[0] or "image/png"
        if len(data.encode("ascii")) <= ANTHROPIC_MAX_IMAGE_BASE64_BYTES or max_pixels == ANTHROPIC_IMAGE_PIXEL_CAPS[-1]:
            return {"type": "base64", "media_type": media_type, "data": data}


def _anthropic_blocks(content: Content) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content or " "}]
    blocks: list[dict[str, Any]] = []
    for block in content:
        if isinstance(block, str) and block:
            blocks.append({"type": "text", "text": block})
        elif isinstance(block, ImageContentBlock):
            blocks.append({"type": "image", "source": _anthropic_image(block)})
        elif not isinstance(block, str):
            raise NotImplementedError(f"unsupported Anthropic content block: {type(block)}")
    return blocks or [{"type": "text", "text": " "}]


def _messages_to_anthropic(messages: list[ChatMessage]) -> tuple[str | None, list[dict[str, Any]]]:
    systems: list[str] = []
    out: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, SystemMessage):
            systems.append(llm._content_text(message.content))
        elif isinstance(message, UserMessage):
            out.append({"role": "user", "content": _anthropic_blocks(message.content)})
        elif isinstance(message, ToolMessage):
            blocks = _anthropic_blocks(message.content)
            single_text = len(blocks) == 1 and blocks[0].get("type") == "text"
            result: dict[str, Any] = {"type": "tool_result", "tool_use_id": message.tool_call_id or "",
                                      "content": str(blocks[0].get("text") or "") if single_text else blocks}
            if not message.success:
                result["is_error"] = True
            out.append({"role": "user", "content": [result]})
        elif isinstance(message, AssistantMessage):
            content_blocks: list[dict[str, Any]] = []
            if message.reasoning and message.reasoning.signature:
                content_blocks.append({"type": "thinking", "thinking": message.reasoning.content,
                                       "signature": message.reasoning.signature})
            if llm._content_text(message.content):
                content_blocks.extend(_anthropic_blocks(message.content))
            for call in message.tool_calls:
                content_blocks.append({"type": "tool_use", "id": call.tool_call_id or "", "name": call.name,
                                       "input": _safe_json_loads(call.arguments)})
            out.append({"role": "assistant", "content": content_blocks or [{"type": "text", "text": " "}]})
        else:
            raise NotImplementedError(f"unsupported Anthropic message: {type(message)}")
    return "\n\n".join(part for part in systems if part) or None, out


def _extract_anthropic(data: dict[str, Any]) -> tuple[str, list[ToolCall], Reasoning | None]:
    text: list[str] = []
    calls: list[ToolCall] = []
    reasoning = None
    for block in data.get("content") or []:
        if block.get("type") == "text":
            text.append(str(block.get("text") or ""))
        elif block.get("type") == "tool_use":
            calls.append(ToolCall(name=str(block.get("name") or ""),
                                  arguments=json.dumps(block.get("input") or {}, ensure_ascii=False),
                                  tool_call_id=block.get("id")))
        elif block.get("type") == "thinking":
            reasoning = Reasoning(signature=block.get("signature"),
                                  content=str(block.get("thinking") or block.get("summary") or ""))
    return "".join(text), calls, reasoning


def _retry_delay(response: httpx.Response | None, attempt: int) -> float:
    if response is not None and response.status_code == 429:
        try:
            return min(120.0, max(1.0, float(response.headers.get("retry-after", ""))))
        except ValueError:
            return min(120.0, 15.0 * (2 ** (attempt - 1)))
    return min(10.0, float(2 ** (attempt - 1)))


class VertexClaudeClient(LLMClient):
    """Anthropic Messages through Vertex AI rawPredict, as a Stirrup LLMClient."""

    def __init__(self, profile: dict[str, Any], log_path: Path) -> None:
        self._profile = profile
        self._project = os.environ["VERTEX_PROJECT"]
        self._log_path = log_path
        self._tokens = VertexToken()

    @property
    def model_slug(self) -> str:
        return self._profile["model"]

    @property
    def max_tokens(self) -> int:
        return self._profile["max_tokens"]

    async def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = (f"https://aiplatform.googleapis.com/v1/projects/{self._project}/locations/{self._profile['location']}"
               f"/publishers/anthropic/models/{self.model_slug}:rawPredict")
        last_error = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            started = time.time()
            response = None
            headers = {"Authorization": f"Bearer {self._tokens.get()}", "Content-Type": "application/json"}
            try:
                async with httpx.AsyncClient(timeout=self._profile["request_timeout"]) as client:
                    response = await client.post(url, headers=headers, json=payload)
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                last_error = repr(exc)
                llm.append_jsonl(self._log_path, {"kind": "anthropic_vertex_messages", "attempt": attempt,
                                                  "error": last_error})
            else:
                text = response.text
                try:
                    body = response.json()
                except ValueError:
                    body = {"raw_text": text}
                llm.append_jsonl(self._log_path, {
                    "kind": "anthropic_vertex_messages", "attempt": attempt, "status_code": response.status_code,
                    "latency_seconds": round(time.time() - started, 3), "request": payload, "response": body})
                if response.status_code < 400:
                    return body
                if response.status_code == 401:     # expired or revoked token: fetch a fresh one
                    self._tokens.invalidate()
                if response.status_code == 400 and any(term in text.lower()
                                                       for term in ("context", "token", "too long", "exceeds")):
                    raise ContextOverflowError(text)
                last_error = text
                if response.status_code not in RETRYABLE_STATUS:
                    break
            if attempt < MAX_ATTEMPTS:
                await asyncio.sleep(_retry_delay(response, attempt))
        raise RuntimeError(f"Anthropic Vertex request failed for {self.model_slug}: {last_error}")

    async def generate(self, messages: list[ChatMessage], tools: dict[str, Tool[Any, Any]]) -> AssistantMessage:
        system, converted = _messages_to_anthropic(messages)
        payload: dict[str, Any] = {"anthropic_version": "vertex-2023-10-16", "max_tokens": self.max_tokens,
                                   "temperature": self._profile["temperature"], "messages": converted}
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = [{"name": tool.name, "description": tool.description,
                                 "input_schema": llm._parameters(tool)} for tool in tools.values()]
        payload["thinking"] = self._profile["thinking"]
        payload["output_config"] = {"effort": self._profile["effort"]}
        started = time.perf_counter()
        data = await self._post(payload)
        ended = time.perf_counter()
        stop_reason = str(data.get("stop_reason") or "")
        if stop_reason in {"max_tokens", "model_context_window_exceeded"}:
            raise ContextOverflowError(f"Anthropic Vertex stop_reason={stop_reason}")
        text, calls, reasoning = _extract_anthropic(data)
        usage = data.get("usage") or {}
        output = int(usage.get("output_tokens") or 0)
        thinking = int((usage.get("output_tokens_details") or {}).get("thinking_tokens") or 0)
        return AssistantMessage(
            reasoning=reasoning, content=text, tool_calls=calls,
            token_usage=TokenUsage(input=int(usage.get("input_tokens") or 0), answer=max(0, output - thinking),
                                   reasoning=thinking),
            metadata={"stop_reason": stop_reason, "stop_sequence": data.get("stop_sequence"),
                      "stop_details": data.get("stop_details"), "response_id": data.get("id")},
            request_start_time=started, request_end_time=ended)


def make_client(profile: dict[str, Any], log_path: Path) -> LLMClient:
    """A judge member's client; profile is its configs/judges/gdpval_<member>.json."""
    if profile["provider"] == "vertex_claude":
        return VertexClaudeClient(profile, log_path)
    return llm.stirrup_client(llm.ModelProfile(**{key: value for key, value in profile.items()
                                                  if not key.startswith("_")}), log_path)
