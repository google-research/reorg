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
"""Model layer with fake HTTP transports and a scripted model: request payloads, structured
calls, sandboxed code execution, the finish tool and web search. No API calls."""

import asyncio
import contextlib
import io
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

import httpx
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stirrup import (AssistantMessage, ContextOverflowError, ImageContentBlock, LLMClient,  # noqa: E402
                     SystemMessage, ToolCall, ToolMessage, UserMessage)
from stirrup.core.models import Reasoning  # noqa: E402

from reorg import llm, web_search, worker  # noqa: E402
from reorg.prompts import PROMPTS  # noqa: E402

GEMINI, QWEN = llm.load_profile("gemini_3_1_pro"), llm.load_profile("qwen_3_6_27b")
RETRIEVAL_SCHEMA = json.loads((PROMPTS / "planner/later_round_retrieval.schema.json").read_text())


@contextlib.contextmanager
def fake_http(*replies):
    """Serve scripted replies (a body, or (status, body)); yield the recorded requests."""
    sent, queue, real = [], list(replies), httpx.Client

    def handler(request):
        sent.append({"url": str(request.url), "headers": request.headers, "json": json.loads(request.content)})
        status, body = queue.pop(0) if isinstance(queue[0], tuple) else (200, queue.pop(0))
        return httpx.Response(status, json=body)

    transport = httpx.MockTransport(handler)
    with mock.patch.object(httpx, "Client", lambda **kwargs: real(transport=transport, **kwargs)), \
            mock.patch.object(llm, "RETRY_DELAY", 0), \
            mock.patch.dict(os.environ, {"GEMINI_API_KEY": "test-key", "VLLM_BASE_URL": "http://vllm.test/v1"}):
        yield sent


def _png() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), (255, 0, 0)).save(buffer, format="PNG")
    return buffer.getvalue()


def _messages():
    return [SystemMessage(content="system text"),
            UserMessage(content=["look at this", ImageContentBlock(data=_png())]),
            AssistantMessage(content="checking", reasoning=Reasoning(signature="sig-text", content=""),
                             tool_calls=[ToolCall(name="finish", arguments='{"reason": "r", "paths": ["a.md"]}',
                                                  tool_call_id="c1", signature="sig-call")]),
            ToolMessage(content="ERROR: file does not exist: a.md", tool_call_id="c1", name="finish", success=False)]


TOOLS = {"finish": worker.finish_tool(["a.md"]), "abandon_task_finish": worker.ABANDON_TOOL}


def _gemini_reply(parts, finish_reason="STOP"):
    return {"candidates": [{"content": {"role": "model", "parts": parts}, "finishReason": finish_reason}],
            "usageMetadata": {"promptTokenCount": 7, "candidatesTokenCount": 3, "thoughtsTokenCount": 2}}


def _vllm_reply(content, finish_reason="stop", **message):
    return {"id": "r", "choices": [{"finish_reason": finish_reason, "message": {"content": content, **message}}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 5, "completion_tokens_details": {"reasoning_tokens": 2}}}


# ---- profiles and agent requests ---------------------------------------------

def test_profiles():
    assert (GEMINI.provider, GEMINI.model, GEMINI.thinking_level, GEMINI.temperature) == \
        ("gemini", "gemini-3.1-pro-preview", "high", 1.0)
    assert (QWEN.provider, QWEN.model, QWEN.enable_thinking, QWEN.temperature, QWEN.top_p, QWEN.top_k) == \
        ("vllm", "Qwen/Qwen3.6-27B", True, 0.6, 0.95, 20)
    assert GEMINI.max_tokens == QWEN.max_tokens == 65536 and GEMINI.request_timeout == QWEN.request_timeout == 1800


def test_gemini_agent_request_and_reply():
    reply = _gemini_reply([{"text": "hidden", "thought": True}, {"text": "Done.", "thoughtSignature": "sig-3"},
                           {"functionCall": {"name": "abandon_task_finish", "args": {"reason": "x"}, "id": "c2"},
                            "thoughtSignature": "sig-4"}])
    with fake_http(reply, _gemini_reply([{"text": "cut"}], "MAX_TOKENS")) as sent:
        client = llm.stirrup_client(GEMINI, None)
        message = asyncio.run(client.generate(_messages(), TOOLS))
        try:
            asyncio.run(client.generate(_messages(), TOOLS))
            raise AssertionError("MAX_TOKENS must surface as a context overflow")
        except ContextOverflowError:
            pass
    request = sent[0]
    assert request["url"] == f"{llm.GEMINI_API}/gemini-3.1-pro-preview:generateContent"
    assert request["headers"]["x-goog-api-key"] == "test-key"
    body = request["json"]
    assert list(body) == ["contents", "generationConfig", "systemInstruction", "tools", "toolConfig"]
    assert body["generationConfig"] == {"maxOutputTokens": 65536, "temperature": 1.0,
                                        "thinkingConfig": {"thinkingLevel": "high"}}
    assert body["systemInstruction"] == {"parts": [{"text": "system text"}]}
    user, model, tool = body["contents"]
    assert user["parts"][1]["inlineData"]["mimeType"] == "image/png"
    assert model == {"role": "model", "parts": [
        {"text": "checking", "thoughtSignature": "sig-text"},
        {"functionCall": {"name": "finish", "args": {"reason": "r", "paths": ["a.md"]}, "id": "c1"},
         "thoughtSignature": "sig-call"}]}
    assert tool["parts"] == [{"functionResponse": {"name": "finish", "id": "c1", "response": {
        "success": False, "content": "ERROR: file does not exist: a.md"}}}]
    declarations = body["tools"][0]["functionDeclarations"]
    assert [d["name"] for d in declarations] == ["finish", "abandon_task_finish"]
    assert declarations[0]["parameters"]["required"] == ["reason", "paths"] and "title" not in json.dumps(declarations)
    assert body["toolConfig"] == {"functionCallingConfig": {"mode": "AUTO"}}
    assert message.content == "Done." and message.reasoning.signature == "sig-3"
    assert (message.tool_calls[0].name, message.tool_calls[0].signature) == ("abandon_task_finish", "sig-4")
    assert (message.token_usage.input, message.token_usage.answer, message.token_usage.reasoning) == (7, 3, 2)


def test_vllm_agent_request_and_reply():
    reply = _vllm_reply("<think>plan</think>Answer", finish_reason="tool_calls", tool_calls=[
        {"id": "c2", "type": "function", "function": {"name": "abandon_task_finish", "arguments": '{"reason": "x"}'}}])
    with fake_http(reply, _vllm_reply("cut", "length")) as sent:
        client = llm.stirrup_client(QWEN, None)
        message = asyncio.run(client.generate(_messages(), TOOLS))
        try:
            asyncio.run(client.generate(_messages(), TOOLS))
            raise AssertionError("length must surface as a context overflow")
        except ContextOverflowError:
            pass
    assert sent[0]["url"] == "http://vllm.test/v1/chat/completions" and "x-goog-api-key" not in sent[0]["headers"]
    body = sent[0]["json"]
    assert {key: body[key] for key in ("model", "max_tokens", "temperature", "top_p", "top_k", "min_p",
                                       "presence_penalty", "repetition_penalty", "chat_template_kwargs",
                                       "tool_choice")} == {
        "model": "Qwen/Qwen3.6-27B", "max_tokens": 65536, "temperature": 0.6, "top_p": 0.95, "top_k": 20,
        "min_p": 0.0, "presence_penalty": 0.0, "repetition_penalty": 1.0,
        "chat_template_kwargs": {"enable_thinking": True}, "tool_choice": "auto"}
    system, user, assistant, tool = body["messages"]
    assert system == {"role": "system", "content": "system text"}
    assert user["content"][0] == {"type": "text", "text": "look at this"}
    assert user["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert assistant["tool_calls"] == [{"id": "c1", "type": "function", "function": {
        "name": "finish", "arguments": '{"reason": "r", "paths": ["a.md"]}'}}]
    assert tool == {"role": "tool", "content": "ERROR: file does not exist: a.md", "tool_call_id": "c1",
                    "name": "finish"}
    assert [t["function"]["name"] for t in body["tools"]] == ["finish", "abandon_task_finish"]
    assert message.content == "Answer" and message.reasoning.content == "plan"
    assert message.tool_calls[0].arguments == '{"reason": "x"}'
    assert (message.token_usage.answer, message.token_usage.reasoning) == (3, 2)


def test_http_retries_429_and_5xx_only():
    with fake_http((429, {}), (503, {}), _gemini_reply([{"text": "ok"}])) as sent:
        assert asyncio.run(llm.stirrup_client(GEMINI, None).generate(_messages(), {})).content == "ok"
    assert len(sent) == 3
    with fake_http((403, {"error": "denied"})) as sent:
        try:
            llm.gemini_generate("m", {}, timeout=1, log_path=None)
            raise AssertionError("403 must not be retried")
        except RuntimeError:
            assert len(sent) == 1
    with fake_http((400, {"error": "input token count exceeds the maximum"})):
        try:
            llm.gemini_generate("m", {}, timeout=1, log_path=None)
            raise AssertionError("an overlong prompt must surface as a context overflow")
        except ContextOverflowError:
            pass


# ---- structured calls --------------------------------------------------------

def _retrieval(first="leave_unchanged"):
    return {"analysis": "a", "recommendations": [{"option_id": first, "reason": "r"}]
            + [{"option_id": f"flow_{i:03d}", "reason": "r"} for i in range(1, 10)]}


def test_vllm_strips_contains_and_python_enforces_it():
    with tempfile.TemporaryDirectory() as tmp, \
            fake_http(_vllm_reply(json.dumps(_retrieval("flow_010"))), _vllm_reply(json.dumps(_retrieval()))) as sent:
        value = llm.complete_json(QWEN, system="s", prompt="p", schema=RETRIEVAL_SCHEMA, temperature=0.2,
                                  log_path=Path(tmp) / "calls.jsonl")
        assert "items match 'contains'" in (Path(tmp) / "calls.jsonl").read_text()
    assert value == _retrieval() and len(sent) == 2      # the first reply had no leave_unchanged option
    body = sent[0]["json"]
    sent_schema = body["response_format"]["json_schema"]["schema"]
    assert not {"contains", "minContains", "maxContains"} & set(sent_schema["properties"]["recommendations"])
    assert sent_schema["properties"]["recommendations"]["items"]["properties"]["option_id"]["pattern"]
    assert body["response_format"]["json_schema"]["strict"] is True
    assert (body["temperature"], body["top_k"], body["chat_template_kwargs"]) == (0.2, 20, {"enable_thinking": True})
    assert body["messages"] == [{"role": "system", "content": "s"}, {"role": "user", "content": "p"}]


def test_complete_json_retries_invalid_json_and_rejected_replies():
    schema = {"type": "object", "required": ["score"], "properties": {"score": {"type": "integer", "minimum": 1}}}
    replies = [_gemini_reply([{"text": "not json"}]), _gemini_reply([{"text": '```json\n{"score": 2}\n```'}]),
               _gemini_reply([{"text": "thinking", "thought": True}, {"text": '{"score": 9}'}])]

    def validate(value):
        if value["score"] < 5:
            raise ValueError("score too low")

    with fake_http(*replies) as sent:
        assert llm.complete_json(GEMINI, system="s", prompt="p", schema=schema, validate=validate) == {"score": 9}
    config = sent[0]["json"]["generationConfig"]
    assert config == {"maxOutputTokens": 65536, "thinkingConfig": {"thinkingLevel": "high"},
                      "responseMimeType": "application/json",
                      "responseSchema": {"type": "OBJECT", "required": ["score"],
                                         "properties": {"score": {"type": "INTEGER", "minimum": 1}}},
                      "temperature": 1.0}
    assert sent[0]["json"]["systemInstruction"] == {"parts": [{"text": "s"}]} and len(sent) == 3
    with fake_http(_gemini_reply([{"text": "[]"}]), _gemini_reply([{"text": '{"score": 0}'}])):
        try:
            llm.complete_json(GEMINI, system="s", prompt="p", schema=schema, attempts=2)
            raise AssertionError("invalid replies must exhaust the attempts")
        except ValueError as exc:
            assert "attempt 2" in str(exc)


def test_schema_check():
    llm.check_schema(_retrieval(), RETRIEVAL_SCHEMA)
    for bad in ({**_retrieval(), "extra": 1}, {**_retrieval(), "analysis": ""},
                {"analysis": "a", "recommendations": _retrieval()["recommendations"][:9]},
                {"analysis": "a", "recommendations": [{"option_id": "flow_1", "reason": "r"}] * 10},
                {"analysis": "a", "recommendations": [{"option_id": "leave_unchanged", "reason": "r"}] * 10}):
        try:
            llm.check_schema(bad, RETRIEVAL_SCHEMA)
            raise AssertionError(f"accepted {bad}")
        except ValueError:
            pass
    ranking = json.loads((PROMPTS / "learning/artifact_ranking.schema.json").read_text())
    llm.check_schema({"group_id": "g", "ranking_tiers": [["artifact_A"]], "hard_invalid": [],
                      "decisive_difference": "d"}, ranking)
    for tiers in ([["artifact_a"]], [["artifact_A", "artifact_A"]], []):     # $ref pattern, uniqueItems, minItems
        assert not llm._matches({"group_id": "g", "ranking_tiers": tiers, "hard_invalid": [],
                                 "decisive_difference": "d"}, ranking, ranking)
    assert not llm._matches(True, {"type": "integer"}, {})


# ---- sandbox, finish tool, web search ----------------------------------------

def test_code_exec_is_confined_to_the_sandbox():
    async def session(exec_env):
        async with exec_env:
            here = await exec_env.run_command("pwd; echo hi > out.txt; echo key=$GEMINI_API_KEY")
            outside = await exec_env.run_command("cat /etc/hosts")
            try:
                exec_env._resolve_and_validate_path("../outside.txt")
                escaped = True
            except ValueError:
                escaped = False
            return here, outside, escaped, await exec_env.file_exists("out.txt")

    with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {"GEMINI_API_KEY": "secret"}):
        box = Path(tmp) / "sandbox"
        here, outside, escaped, found = asyncio.run(session(worker.WorkspaceCodeExec(box)))
        assert Path(here.stdout.splitlines()[0]).resolve() == box.resolve()
        assert (box / "out.txt").read_text() == "hi\n" and found and box.is_dir()   # the sandbox outlives the session
        assert "key=\n" in here.stdout                                             # no API key in the agent's shell
        assert outside.error_kind == "absolute_path_detected" and not escaped


class ScriptedClient(LLMClient):
    """Calls the given tools in order and records the message each call answers."""

    model_slug, max_tokens = "scripted", 65536

    def __init__(self, calls):
        self.calls, self.seen = list(calls), []

    async def generate(self, messages, tools):
        self.seen.append(messages[-1])
        name, args = self.calls.pop(0)
        return AssistantMessage(content="", tool_calls=[ToolCall(name=name, arguments=json.dumps(args),
                                                                 tool_call_id=f"c{len(self.seen)}")])


def test_finish_tool_checks_the_expected_paths():
    calls = [("finish", {"reason": "r", "paths": ["current_artifact/a.md", "scratch/notes.txt"]}),
             ("finish", {"reason": "r", "paths": ["current_artifact"]}),
             ("finish", {"reason": "r", "paths": ["../a.md"]}),
             ("finish", {"reason": "r", "paths": ["current_artifact/missing.md"]}),
             ("finish", {"reason": "r", "paths": ["current_artifact/a.md"]})]
    client = ScriptedClient(calls)
    with tempfile.TemporaryDirectory() as tmp, mock.patch.object(worker, "stirrup_client", lambda *_: client), \
            contextlib.redirect_stdout(io.StringIO()):
        box, logs = Path(tmp) / "sandbox", Path(tmp) / "logs"
        (box / "current_artifact").mkdir(parents=True)
        (box / "current_artifact" / "a.md").write_text("done\n")
        run = worker.run_agent(GEMINI, system_prompt="s", prompt="p", tools=[worker.WorkspaceCodeExec(box)],
                               finish_tools=[worker.finish_tool(["deliverable.md"]), worker.ABANDON_TOOL], logs=logs)
        assert sorted(path.name for path in logs.iterdir()) == ["finish.json", "history.json", "metadata.json"]
    errors = [message.content for message in client.seen[1:]]
    assert errors[0].startswith("ERROR: submit exactly 1 final deliverable file(s)")
    assert "submitted file extensions must match expected deliverables: expected {'.md': 1}" in errors[0]
    assert "path is a directory, not a file: current_artifact" in errors[1]
    assert errors[2] == "ERROR: path must be relative and stay inside the task workspace: ../a.md"
    assert errors[3] == "ERROR: file does not exist: current_artifact/missing.md"
    assert run.finish == {"reason": "r", "paths": ["current_artifact/a.md"]} and run.finish_type == "FinishParams"
    assert run.usage == {"input": 0, "answer": 0, "reasoning": 0} and run.history


def test_abandon_ends_the_session():
    client = ScriptedClient([("abandon_task_finish", {"reason": "inputs missing"})])
    with tempfile.TemporaryDirectory() as tmp, mock.patch.object(worker, "stirrup_client", lambda *_: client), \
            contextlib.redirect_stdout(io.StringIO()):
        run = worker.run_agent(QWEN, system_prompt="s", prompt="p", tools=[worker.WorkspaceCodeExec(Path(tmp))],
                               finish_tools=[worker.finish_tool([]), worker.ABANDON_TOOL], logs=Path(tmp) / "logs")
    assert (run.finish, run.finish_type) == ({"reason": "inputs missing"}, "AbandonParams")


def test_web_search_request_and_guards():
    grounded = {"candidates": [{"content": {"parts": [{"text": "Setbacks vary by state."}]}, "groundingMetadata": {
        "webSearchQueries": ["well setback rules"],
        "groundingChunks": [{"web": {"title": "EPA private wells", "uri": "https://epa.gov/wells"}},
                            {"web": {"title": "GDPval task copy", "uri": "https://example.com/t"}}],
        "groundingSupports": [{"segment": {"text": "Setbacks vary."}, "groundingChunkIndices": [0]}]}}]}
    tools = web_search.worker_web_tools(QWEN, task_text='Review the "Oakdale municipal well permit".',
                                        log_path=None)
    search = tools[1]

    async def queries():
        tool = await search.__aenter__()
        return [await tool.executor(web_search.SearchParams(query=query)) for query in (
            "gdpval municipal well answer", '"Oakdale municipal well permit"', "well setback rules", "more", "again")]

    with fake_http(grounded, grounded) as sent:      # four uses per action, blocked queries included
        benchmark, quoted, found, _, limited = asyncio.run(queries())
    assert len(sent) == 2 and sent[0]["url"] == f"{llm.GEMINI_API}/gemini-3.5-flash:generateContent"
    assert sent[0]["json"]["tools"] == [{"google_search": {}}]
    assert sent[0]["json"]["generationConfig"] == {"maxOutputTokens": 2048, "thinkingConfig": {"thinkingLevel": "low"}}
    assert not benchmark.success and "cannot be used to search for GDPval" in benchmark.content
    assert not quoted.success and "quoted task-prompt text" in quoted.content
    assert found.success and "https://epa.gov/wells" in found.content and "GDPval" not in found.content
    assert not limited.success and "use limit reached" in limited.content
    gemini_search = web_search.worker_web_tools(GEMINI, task_text="t", log_path=None)[1]
    assert (gemini_search._model, gemini_search._temperature) == ("gemini-3.1-pro-preview", 0.0)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
