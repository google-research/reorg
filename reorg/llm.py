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
"""Model access for the worker W, the stakeholders, the planner and the judges (App. A.2).

Two providers, selected by a model profile (configs/models/<name>.json):

    gemini   Gemini Developer API, key from GEMINI_API_KEY
    vllm     an OpenAI-compatible vLLM server at VLLM_BASE_URL (its native endpoint, ending in /v1)

`stirrup_client` serves agent sessions; `complete_json` serves single structured calls.
Request payloads carry the profile's generation and thinking config and sampling
fields, the tool schemas and image parts. Every request is retried on 408/429/5xx
and transport errors with exponential backoff.
"""

from __future__ import annotations

import asyncio
import base64
import json
import math
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import httpx
from stirrup import (AssistantMessage, AudioContentBlock, ChatMessage, ContextOverflowError, EmptyParams,
                     ImageContentBlock, LLMClient, SystemMessage, TokenUsage, Tool, ToolCall, ToolMessage,
                     UserMessage, VideoContentBlock)
from stirrup.clients.utils import content_to_openai
from stirrup.core.models import Content, Reasoning

CONFIGS = Path(__file__).resolve().parent.parent / "configs" / "models"
GEMINI_API = "https://generativelanguage.googleapis.com/v1beta/models"
ATTEMPTS = 5                                    # HTTP attempts per request
RETRY_DELAY = 5.0                               # seconds before the first retry; doubles per retry
# Tool-parameter keywords dropped for both providers' function declarations.
TOOL_SCHEMA_DROP = frozenset({"$defs", "$schema", "additionalProperties", "default", "examples", "title"})
# Response-schema keywords outside Gemini's structured-output subset.
GEMINI_SCHEMA_DROP = frozenset({"$comment", "$defs", "$id", "$schema", "additionalProperties", "contains", "default",
                                "examples", "maxContains", "minContains", "minLength", "pattern", "title"})
# Keywords vLLM's guided decoding rejects. Both providers' replies are checked in Python.
VLLM_SCHEMA_DROP = frozenset({"contains", "minContains", "maxContains"})
SAMPLING = ("top_p", "top_k", "min_p", "presence_penalty", "repetition_penalty")
_LOG_LOCK = threading.Lock()


@dataclass(frozen=True)
class ModelProfile:
    name: str
    provider: str                                   # "gemini" | "vllm"
    model: str
    max_tokens: int = 65536
    temperature: float | None = None
    thinking_level: str | None = None               # gemini: "high" | "low"
    enable_thinking: bool = False                   # vllm chat-template thinking (Qwen)
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    presence_penalty: float | None = None
    repetition_penalty: float | None = None
    images: bool = True                             # the model accepts image inputs
    request_timeout: float = 1800.0                 # seconds per model request


def load_profile(name: str) -> ModelProfile:
    """configs/models/<name>.json, or a path to such a file."""
    path = Path(name) if name.endswith(".json") else CONFIGS / f"{name}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ModelProfile(**{key: value for key, value in payload.items() if not key.startswith("_")})


def stirrup_client(profile: ModelProfile, log_path: Path) -> Any:
    """A Stirrup LLMClient for agent sessions; every request/response is appended to log_path (JSONL)."""
    return {"gemini": GeminiClient, "vllm": VLLMClient}[profile.provider](profile, log_path)


def complete_json(profile: ModelProfile, *, system: str, prompt: str, schema: dict,
                  temperature: float | None = None, log_path: Path | None = None, attempts: int = 3,
                  validate: Callable[[dict], None] | None = None) -> dict:
    """One structured model call, retried until the reply parses, matches `schema`, and passes
    `validate` (which raises ValueError to reject). `temperature` overrides the profile's."""
    temperature = profile.temperature if temperature is None else temperature
    if profile.provider == "gemini":
        config = {"maxOutputTokens": profile.max_tokens, "thinkingConfig": {"thinkingLevel": profile.thinking_level},
                  "responseMimeType": "application/json",
                  "responseSchema": _gemini_response_schema(_inline_refs(schema, schema))}
        if temperature is not None:
            config["temperature"] = temperature
        payload = {"systemInstruction": {"parts": [{"text": system}]},
                   "contents": [{"role": "user", "parts": [{"text": prompt}]}], "generationConfig": config}
    else:
        payload = {"model": profile.model,
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                   "temperature": temperature, **_sampling(profile), "max_tokens": profile.max_tokens,
                   "response_format": {"type": "json_schema", "json_schema": {
                       "name": "response", "schema": _drop_keys(schema, VLLM_SCHEMA_DROP), "strict": True}}}
        if profile.enable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": True}
    errors = []
    for attempt in range(1, attempts + 1):
        if profile.provider == "gemini":
            data = gemini_generate(profile.model, payload, timeout=profile.request_timeout, log_path=log_path)
            text = "".join(str(part.get("text") or "") for candidate in data.get("candidates") or []
                           for part in (candidate.get("content") or {}).get("parts") or [] if not part.get("thought"))
        else:
            data = _vllm_chat(payload, timeout=profile.request_timeout, log_path=log_path)
            text = str(((data.get("choices") or [{}])[0].get("message") or {}).get("content") or "")
        try:
            value = _parse_object(text)
            check_schema(value, schema)
            if validate:
                validate(value)
            return value
        except ValueError as exc:                   # json.JSONDecodeError is a ValueError
            errors.append(f"attempt {attempt}: {exc}")
            append_jsonl(log_path, {"kind": "invalid_reply", "attempt": attempt, "error": str(exc)})
    raise ValueError("no valid structured reply: " + " | ".join(errors))


# ---- HTTP --------------------------------------------------------------------

def gemini_generate(model: str, payload: dict, *, timeout: float, log_path: Path | None,
                    kind: str = "gemini_generate_content") -> dict:
    headers = {"Content-Type": "application/json", "Accept-Encoding": "identity",
               "x-goog-api-key": os.environ["GEMINI_API_KEY"]}
    return _post(f"{GEMINI_API}/{model}:generateContent", payload, headers, timeout, log_path, kind)


def _vllm_chat(payload: dict, *, timeout: float, log_path: Path | None) -> dict:
    url = os.environ["VLLM_BASE_URL"].rstrip("/") + "/chat/completions"
    return _post(url, payload, {"Content-Type": "application/json"}, timeout, log_path, "vllm_chat_completions")


def _post(url: str, payload: dict, headers: dict, timeout: float, log_path: Path | None, kind: str) -> dict:
    error = ""
    for attempt in range(1, ATTEMPTS + 1):
        if attempt > 1:
            time.sleep(RETRY_DELAY * 2 ** (attempt - 2))
        started = time.time()
        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.post(url, headers=headers, json=payload)
        except httpx.TransportError as exc:
            error = repr(exc)
            append_jsonl(log_path, {"kind": kind, "url": url, "attempt": attempt, "error": error})
            continue
        try:
            body = response.json()
        except ValueError:
            body = {"raw_text": response.text}
        append_jsonl(log_path, {"kind": kind, "url": url, "attempt": attempt, "status_code": response.status_code,
                                "latency_seconds": round(time.time() - started, 3), "request": payload,
                                "response": body})
        if response.status_code < 400:
            return body
        error = response.text
        # A prompt over the context window: Stirrup drops a completed turn and retries.
        if response.status_code == 400 and any(term in error.lower()
                                               for term in ("context", "token", "too long", "exceeds")):
            raise ContextOverflowError(error)
        if response.status_code not in (408, 429) and response.status_code < 500:
            break
    raise RuntimeError(f"{kind} failed: {error}")


def append_jsonl(path: Path | None, row: dict) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with _LOG_LOCK, path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---- schemas -----------------------------------------------------------------

def _inline_refs(value: Any, root: dict) -> Any:
    """Replace local "$ref"s ("#/$defs/...") by their targets; sibling keys override the target's."""
    if isinstance(value, dict):
        if "$ref" in value:
            target = root
            for part in value["$ref"].removeprefix("#/").split("/"):
                target = target[part]
            return _inline_refs({**target, **{k: v for k, v in value.items() if k != "$ref"}}, root)
        return {key: _inline_refs(child, root) for key, child in value.items()}
    if isinstance(value, list):
        return [_inline_refs(item, root) for item in value]
    return value


def _tool_schema(value: Any) -> Any:
    """Pydantic tool parameters -> the function-declaration subset both providers accept."""
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, child in value.items():
            if key in TOOL_SCHEMA_DROP or key == "$ref":
                continue
            if key == "anyOf":                  # Optional[X] -> X, nullable
                options = [_tool_schema(item) for item in child if item.get("type") != "null"]
                if options:
                    cleaned.update(options[0])
                    if len(options) != len(child):
                        cleaned["nullable"] = True
                continue
            cleaned[key] = _tool_schema(child)
        if "properties" in cleaned and "type" not in cleaned:
            cleaned["type"] = "object"
        if cleaned.get("type") == "array" and "items" not in cleaned:
            cleaned["items"] = {}
        return cleaned
    if isinstance(value, list):
        return [_tool_schema(item) for item in value]
    return value


def _parameters(tool: Tool) -> dict:
    if tool.parameters is EmptyParams:
        return {"type": "object", "properties": {}}
    schema = tool.parameters.model_json_schema()
    parameters = _tool_schema(_inline_refs(schema, schema))
    parameters.setdefault("type", "object")
    parameters.setdefault("properties", {})
    return parameters


def _gemini_response_schema(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            if key in GEMINI_SCHEMA_DROP or key == "$ref":
                continue
            if key == "const":
                cleaned["enum"] = [item]
                continue
            cleaned[key] = str(item).upper() if key == "type" else _gemini_response_schema(item)
        return cleaned
    if isinstance(value, list):
        return [_gemini_response_schema(item) for item in value]
    return value


def _drop_keys(value: Any, keys: frozenset[str]) -> Any:
    if isinstance(value, dict):
        return {key: _drop_keys(child, keys) for key, child in value.items() if key not in keys}
    if isinstance(value, list):
        return [_drop_keys(item, keys) for item in value]
    return value


_TYPES = {"object": dict, "array": list, "string": str, "integer": int, "number": (int, float),
          "boolean": bool, "null": type(None)}


def check_schema(value: Any, schema: dict, root: dict | None = None, where: str = "reply") -> None:
    """The JSON Schema subset of reorg/prompts/**/*.schema.json; raises ValueError at the first violation."""
    root = schema if root is None else root
    if "$ref" in schema:
        return check_schema(value, _inline_refs(schema, root), root, where)
    kind = schema.get("type")
    if kind and (not isinstance(value, _TYPES[kind]) or (kind in ("integer", "number") and isinstance(value, bool))):
        raise ValueError(f"{where}: expected {kind}")
    if "const" in schema and value != schema["const"]:
        raise ValueError(f"{where}: expected {schema['const']!r}")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{where}: {value!r} not in {schema['enum']}")
    if isinstance(value, str):
        if not schema.get("minLength", 0) <= len(value) <= schema.get("maxLength", math.inf):
            raise ValueError(f"{where}: string length {len(value)} out of range")
        if "pattern" in schema and not re.search(schema["pattern"], value):
            raise ValueError(f"{where}: {value!r} does not match {schema['pattern']}")
    if isinstance(value, int | float) and not isinstance(value, bool):
        if not schema.get("minimum", -math.inf) <= value <= schema.get("maximum", math.inf):
            raise ValueError(f"{where}: {value} out of range")
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        missing = [key for key in schema.get("required", ()) if key not in value]
        extra = [key for key in value if key not in properties] if schema.get("additionalProperties") is False else []
        if missing or extra:
            raise ValueError(f"{where}: missing {missing}, unexpected {extra}")
        for key, child in value.items():
            if key in properties:
                check_schema(child, properties[key], root, f"{where}.{key}")
    if isinstance(value, list):
        if not schema.get("minItems", 0) <= len(value) <= schema.get("maxItems", math.inf):
            raise ValueError(f"{where}: {len(value)} items out of range")
        if schema.get("uniqueItems") and len({json.dumps(item, sort_keys=True) for item in value}) != len(value):
            raise ValueError(f"{where}: items are not unique")
        for index, item in enumerate(value):
            if "items" in schema:
                check_schema(item, schema["items"], root, f"{where}[{index}]")
        if "contains" in schema:
            count = sum(_matches(item, schema["contains"], root) for item in value)
            if not schema.get("minContains", 1) <= count <= schema.get("maxContains", math.inf):
                raise ValueError(f"{where}: {count} items match 'contains'")


def _matches(value: Any, schema: dict, root: dict) -> bool:
    try:
        check_schema(value, schema, root)
    except ValueError:
        return False
    return True


def _parse_object(text: str) -> dict:
    lines = text.strip().splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    value = json.loads("\n".join(lines).strip())
    if not isinstance(value, dict):
        raise ValueError("reply is not a JSON object")
    return value


def _sampling(profile: ModelProfile) -> dict:
    return {key: getattr(profile, key) for key in SAMPLING if getattr(profile, key) is not None}


# ---- Gemini agent client -----------------------------------------------------

def gemini_generation_config(*, max_output_tokens: int, temperature: float | None,
                             thinking_level: str | None) -> dict:
    config: dict[str, Any] = {"maxOutputTokens": max_output_tokens}
    if temperature is not None:
        config["temperature"] = temperature
    if thinking_level:
        config["thinkingConfig"] = {"thinkingLevel": thinking_level}
    return config


def _text_part(text: str, signature: str | None) -> dict:
    return {"text": text, "thoughtSignature": signature} if signature else {"text": text}


def _gemini_parts(content: Content, signature: str | None = None) -> list[dict]:
    """Text and media parts; a thought signature rides on the first text part."""
    if isinstance(content, str):
        return [_text_part(content, signature)]
    parts = []
    for block in content:
        if isinstance(block, str):
            if block:
                parts.append(_text_part(block, signature))
                signature = None
        elif isinstance(block, ImageContentBlock | AudioContentBlock | VideoContentBlock):
            parts.append({"inlineData": {"mimeType": block.mime_type,
                                         "data": base64.b64encode(block.data).decode("ascii")}})
        else:
            raise NotImplementedError(f"Unsupported content block for Gemini: {type(block)}")
    return parts or [_text_part("", signature)]


def _gemini_message(message: ChatMessage) -> dict:
    if isinstance(message, UserMessage):
        return {"role": "user", "parts": _gemini_parts(message.content)}
    if isinstance(message, ToolMessage):
        content = message.content if isinstance(message.content, str) else "\n".join(map(str, message.content))
        response = {"name": message.name or "tool", "response": {"success": message.success, "content": content}}
        if message.tool_call_id:
            response["id"] = message.tool_call_id
        return {"role": "user", "parts": [{"functionResponse": response}]}
    if isinstance(message, AssistantMessage):
        signature = message.reasoning.signature if message.reasoning else None
        parts = _gemini_parts(message.content, signature) if message.content else []
        for call in message.tool_calls:
            try:
                args = json.loads(call.arguments) if call.arguments else {}
            except json.JSONDecodeError:
                args = {}
            function_call = {"name": call.name, "args": args}
            if call.tool_call_id:
                function_call["id"] = call.tool_call_id
            parts.append({"functionCall": function_call, "thoughtSignature": call.signature}
                         if call.signature else {"functionCall": function_call})
        return {"role": "model", "parts": parts or [{"text": ""}]}
    raise NotImplementedError(f"Unsupported message type for Gemini: {type(message)}")


def gemini_request(profile: ModelProfile, messages: list[ChatMessage], tools: dict[str, Tool]) -> dict:
    system = [part for message in messages if isinstance(message, SystemMessage)
              for part in _gemini_parts(message.content)]
    payload: dict[str, Any] = {
        "contents": [_gemini_message(message) for message in messages if not isinstance(message, SystemMessage)],
        "generationConfig": gemini_generation_config(max_output_tokens=profile.max_tokens,
                                                     temperature=profile.temperature,
                                                     thinking_level=profile.thinking_level)}
    if system:
        payload["systemInstruction"] = {"parts": system}
    if tools:
        payload["tools"] = [{"functionDeclarations": [
            {"name": tool.name, "description": tool.description, "parameters": _parameters(tool)}
            for tool in tools.values()]}]
        payload["toolConfig"] = {"functionCallingConfig": {"mode": "AUTO"}}
    return payload


class GeminiClient(LLMClient):
    """Gemini generateContent with function calling, as a Stirrup LLMClient."""

    def __init__(self, profile: ModelProfile, log_path: Path | None) -> None:
        self.profile = profile
        self.log_path = log_path

    @property
    def model_slug(self) -> str:
        return self.profile.model

    @property
    def max_tokens(self) -> int:
        return self.profile.max_tokens

    async def generate(self, messages: list[ChatMessage], tools: dict[str, Tool]) -> AssistantMessage:
        payload = gemini_request(self.profile, messages, tools)
        started = time.perf_counter()
        data = await asyncio.to_thread(gemini_generate, self.profile.model, payload,
                                       timeout=self.profile.request_timeout, log_path=self.log_path)
        ended = time.perf_counter()
        candidates = data.get("candidates") or []
        if not candidates:
            raise RuntimeError(f"Gemini response had no candidates: {data}")
        candidate = candidates[0]
        text, calls, reasoning = [], [], None
        for part in (candidate.get("content") or {}).get("parts") or []:
            signature = part.get("thoughtSignature") or part.get("thought_signature")
            if "functionCall" in part:
                call = part["functionCall"]
                calls.append(ToolCall(signature=signature, name=str(call.get("name") or ""),
                                      arguments=json.dumps(call.get("args") or {}, ensure_ascii=False),
                                      tool_call_id=call.get("id")))
            elif "text" in part and not part.get("thought"):
                text.append(str(part.get("text") or ""))
                if signature:
                    reasoning = Reasoning(signature=signature, content="")
            elif signature:
                reasoning = Reasoning(signature=signature, content="")
        finish_reason = str(candidate.get("finishReason") or "")
        if finish_reason in {"MAX_TOKENS", "LENGTH"}:
            raise ContextOverflowError(f"Gemini finishReason={finish_reason}")
        usage = data.get("usageMetadata") or {}
        return AssistantMessage(
            reasoning=reasoning, content="".join(text), tool_calls=calls,
            token_usage=TokenUsage(input=int(usage.get("promptTokenCount") or 0),
                                   answer=int(usage.get("candidatesTokenCount") or 0),
                                   reasoning=int(usage.get("thoughtsTokenCount") or 0)),
            metadata={"finish_reason": finish_reason, "safety_ratings": candidate.get("safetyRatings"),
                      "grounding_metadata": candidate.get("groundingMetadata")},
            request_start_time=started, request_end_time=ended)


# ---- vLLM agent client -------------------------------------------------------

def _content_text(content: Content) -> str:
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, ImageContentBlock):
            parts.append("[image content omitted for text-only model]")
        elif isinstance(block, AudioContentBlock):
            parts.append("[audio content omitted for text-only model]")
        elif isinstance(block, VideoContentBlock):
            parts.append("[video content omitted for text-only model]")
        else:
            raise NotImplementedError(f"Unsupported content block: {type(block)}")
    return "\n".join(part for part in parts if part)


def _openai_message(message: ChatMessage) -> dict:
    if isinstance(message, SystemMessage):
        return {"role": "system", "content": _content_text(message.content)}
    if isinstance(message, UserMessage):
        return {"role": "user", "content": content_to_openai(message.content)}
    if isinstance(message, ToolMessage):
        return {"role": "tool", "content": _content_text(message.content), "tool_call_id": message.tool_call_id,
                "name": message.name}
    if isinstance(message, AssistantMessage):
        out: dict[str, Any] = {"role": "assistant", "content": _content_text(message.content) or None}
        if message.tool_calls:
            out["tool_calls"] = [{"id": call.tool_call_id, "type": "function",
                                  "function": {"name": call.name, "arguments": call.arguments}}
                                 for call in message.tool_calls]
        return out
    raise NotImplementedError(f"Unsupported message type for OpenAI chat: {type(message)}")


def vllm_request(profile: ModelProfile, messages: list[ChatMessage], tools: dict[str, Tool]) -> dict:
    payload = {"model": profile.model, "messages": [_openai_message(message) for message in messages],
               "max_tokens": profile.max_tokens, "temperature": profile.temperature, **_sampling(profile)}
    if profile.enable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": True}
    if tools:
        payload["tools"] = [{"type": "function", "function": {"name": tool.name, "description": tool.description,
                                                              "parameters": _parameters(tool)}}
                            for tool in tools.values()]
        payload["tool_choice"] = "auto"
    return payload


class VLLMClient(LLMClient):
    """vLLM's OpenAI-compatible chat completions, as a Stirrup LLMClient."""

    def __init__(self, profile: ModelProfile, log_path: Path | None) -> None:
        self.profile = profile
        self.log_path = log_path

    @property
    def model_slug(self) -> str:
        return self.profile.model

    @property
    def max_tokens(self) -> int:
        return self.profile.max_tokens

    async def generate(self, messages: list[ChatMessage], tools: dict[str, Tool]) -> AssistantMessage:
        payload = vllm_request(self.profile, messages, tools)
        started = time.perf_counter()
        data = await asyncio.to_thread(_vllm_chat, payload, timeout=self.profile.request_timeout,
                                       log_path=self.log_path)
        ended = time.perf_counter()
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"OpenAI-compatible response had no choices: {data}")
        choice = choices[0]
        message = choice.get("message") or {}
        calls = [ToolCall(tool_call_id=call.get("id"), name=str((call.get("function") or {}).get("name") or ""),
                          arguments=str((call.get("function") or {}).get("arguments") or "{}"))
                 for call in message.get("tool_calls") or []]
        content = str(message.get("content") or "")
        if "<|" in content:                     # stray chat-template markers
            content = re.sub(r"<\|[^|]+\|>", "", content).strip()
        think = [match.strip() for match in re.findall(r"<think>(.*?)</think>", content, re.DOTALL | re.IGNORECASE)]
        if think:                               # thinking left inline when the server has no reasoning parser
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL | re.IGNORECASE).strip()
        reasoning = str(message.get("reasoning_content") or message.get("reasoning")
                        or "\n".join(text for text in think if text))
        finish_reason = str(choice.get("finish_reason") or "")
        if finish_reason in {"length", "max_tokens"}:
            raise ContextOverflowError(f"vLLM finish_reason={finish_reason}")
        usage = data.get("usage") or {}
        completion = int(usage.get("completion_tokens") or 0)
        thinking = int((usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0)
        return AssistantMessage(
            reasoning=Reasoning(content=reasoning) if reasoning else None, content=content, tool_calls=calls,
            token_usage=TokenUsage(input=int(usage.get("prompt_tokens") or 0), answer=max(0, completion - thinking),
                                   reasoning=thinking),
            metadata={"finish_reason": finish_reason, "response_id": data.get("id"),
                      "response_model": data.get("model")},
            request_start_time=started, request_end_time=ended)
