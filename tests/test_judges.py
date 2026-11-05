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
"""Evaluation tests (App. A.5): panel vote, score formulas, content cap, status
rules and request shapes. No API calls, no benchmark data, fake upstreams.

    python3 -m pytest -q tests/test_judges.py
"""

import asyncio
import io
import json
import sys
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.gdpval.judge import providers, sandbox, vote                # noqa: E402
from data.gdpval.judge.prompting import (                              # noqa: E402
    JudgeFinishParams, build_judge_prompt, finish_judgment_tool)
from reorg.domain import DOMAINS, load_module  # noqa: E402
from reorg.files import read_json  # noqa: E402

jobbench = load_module(DOMAINS / "jobbench" / "judge.py")
lab = load_module(DOMAINS / "lab" / "judge.py")
CONFIGS = ROOT / "configs" / "judges"


# ---- GDPval panel ----------------------------------------------------------

def _judgment(task_id, candidate, expert, candidate_label="A", status="ok"):
    expert_label = "B" if candidate_label == "A" else "A"
    points = {candidate_label: candidate, expert_label: expert}
    return {"task_id": task_id, "status": status,
            "mapping": {candidate_label: {"kind": "submission"}, expert_label: {"kind": "expert"}},
            "response_a_raw_points": points["A"], "response_b_raw_points": points["B"]}


def test_candidate_outcome_restores_identities():
    assert vote.candidate_outcome(_judgment("t", 10, 8, "A")) == "win"
    assert vote.candidate_outcome(_judgment("t", 10, 8, "B")) == "win"     # candidate shown as Response B
    assert vote.candidate_outcome(_judgment("t", -2, 3, "B")) == "loss"    # negative criteria are penalties
    assert vote.candidate_outcome(_judgment("t", 5, 5, "A")) == "tie"


def test_majority_rules():
    results = {
        "gemini31pro": [_judgment("win2", 9, 1), _judgment("split", 9, 1), _judgment("tie2", 4, 4),
                        _judgment("missing", 9, 1), _judgment("failed", 9, 1)],
        "qwen36": [_judgment("win2", 9, 1, "B"), _judgment("split", 4, 4), _judgment("tie2", 4, 4),
                   _judgment("failed", 9, 1, status="infra_error")],
        "sonnet46": [_judgment("win2", 1, 9), _judgment("split", 1, 9), _judgment("tie2", 9, 1),
                     _judgment("missing", 9, 1), _judgment("failed", 9, 1)],
    }
    decisions = vote.majority(results)
    assert decisions["win2"]["outcome"] == "win" and decisions["win2"]["reason"] is None
    assert decisions["tie2"]["outcome"] == "tie"
    assert decisions["split"] == {"outcome": "tie", "reason": "no_strict_majority",
                                  "votes": {"gemini31pro": "win", "qwen36": "tie", "sonnet46": "loss"}}
    assert decisions["missing"]["outcome"] is None and "qwen36" in decisions["missing"]["reason"]
    assert decisions["failed"]["outcome"] is None


def test_win_tie_denominator_is_the_cohort():
    decisions = {"a": {"outcome": "win"}, "b": {"outcome": "tie"}, "c": {"outcome": "loss"}, "d": {"outcome": None}}
    summary = vote.win_tie(decisions, ["a", "b", "c", "d", "no_submission"])
    assert summary == {"cohort": 5, "win": 1, "tie": 1, "loss": 1, "unavailable": 2, "win_tie_rate": 0.4}


def test_plans_are_seeded_and_blinded(tmp_path):
    submissions = {f"task-{index}": tmp_path / f"task-{index}" for index in range(40)}
    first, again = sandbox.plans(submissions, seed=0), sandbox.plans(submissions, seed=0)
    assert first == again
    assert [plan.attempt_id for plan in first[:2]] == ["000000_task-0", "000001_task-1"]
    kinds = [plan.mapping["A"]["kind"] for plan in first]
    assert set(kinds) == {"submission", "expert"}                         # both visible orders occur
    assert all({plan.mapping["A"]["kind"], plan.mapping["B"]["kind"]} == {"submission", "expert"} for plan in first)


def test_sandbox_layout(tmp_path):
    data = tmp_path / "gdpval"
    for name in ("reference_files/abc/input.xlsx", "deliverable_files/x/report.docx", "deliverable_files/y/report.docx"):
        (data / name).parent.mkdir(parents=True, exist_ok=True)
        (data / name).write_text(name)
    submission = tmp_path / "run" / "t1" / "final" / "submission"
    (submission / "sub").mkdir(parents=True)
    (submission / "memo.md").write_text("memo")
    (submission / "sub" / "table.csv").write_text("a,b")
    task = {"task_id": "t1", "sector": "S", "occupation": "O", "prompt": "Do it.", "rubric_pretty": "+1 x",
            "rubric_json": json.dumps([{"points": 1}]), "reference_files": ["reference_files/abc/input.xlsx"],
            "deliverable_files": ["deliverable_files/x/report.docx", "deliverable_files/y/report.docx"]}
    (plan,) = sandbox.plans({"t1": submission}, seed=0)
    workspace = tmp_path / "workspace"
    sandbox.stage(plan, task, data, workspace, "INSTRUCTION")
    candidate = "A" if plan.mapping["A"]["kind"] == "submission" else "B"
    expert = "B" if candidate == "A" else "A"
    assert read_json(workspace / f"response_{candidate}" / "manifest.json") == {"files": [
        {"path": "files/memo.md", "description": "memo.md"},
        {"path": "files/sub/table.csv", "description": "table.csv"}]}
    assert read_json(workspace / f"response_{expert}" / "manifest.json") == {"files": [
        {"path": "files/report.docx", "description": "report.docx"},
        {"path": "files/report_2.docx", "description": "report_2.docx"}]}
    assert (workspace / "reference_files" / "abc" / "input.xlsx").is_file()
    assert "- `reference_files/abc/input.xlsx`" in (workspace / "task.md").read_text()
    comparison = read_json(workspace / "comparison.json")
    assert comparison["comparison_id"] == plan.attempt_id and comparison["network_tools_enabled"] is False
    assert (workspace / "INSTRUCTION.md").read_text() == "INSTRUCTION\n"
    assert sorted(path.name for path in workspace.iterdir()) == [
        "INSTRUCTION.md", "comparison.json", "reference_files", "response_A", "response_B", "rubric.json",
        "rubric_pretty.md", "scratch", "task.md"]


def test_prompt_and_finish_tool():
    prompt = build_judge_prompt(40)
    assert "You have a hard budget of 40 `code_exec` calls." in prompt
    assert '"tie" when the totals are equal.' in prompt
    tool = finish_judgment_tool()
    good = JudgeFinishParams(analysis="x", response_a_raw_points=3, response_b_raw_points=3, preference="tie")
    bad = JudgeFinishParams(analysis="x", response_a_raw_points=3, response_b_raw_points=1, preference="B")
    assert asyncio.run(tool.executor(good)).success
    assert not asyncio.run(tool.executor(bad)).success


def test_judge_profiles():
    panel = read_json(CONFIGS / "gdpval_trio.json")
    members = {name: read_json(CONFIGS / f"gdpval_{name}.json") for name in panel["members"]}
    assert [profile["provider"] for profile in members.values()] == ["gemini", "vllm", "vertex_claude"]
    assert all(profile["max_tokens"] == 65536 for profile in members.values())
    assert (panel["max_turns"], panel["code_exec_max_uses"], panel["seed"]) == (100, 40, 0)
    assert [members[name]["temperature"] for name in ("gemini31pro", "qwen36", "sonnet46")] == [0.6, 0.6, 1.0]


def test_member_clients(tmp_path, monkeypatch):
    from reorg import llm

    monkeypatch.setenv("VERTEX_PROJECT", "test-project")
    clients = {name: providers.make_client(read_json(CONFIGS / f"gdpval_{name}.json"), tmp_path / "calls.jsonl")
               for name in ("gemini31pro", "qwen36", "sonnet46")}
    assert isinstance(clients["gemini31pro"], llm.GeminiClient)
    assert isinstance(clients["qwen36"], llm.VLLMClient)
    assert isinstance(clients["sonnet46"], providers.VertexClaudeClient)
    assert clients["gemini31pro"].profile.temperature == 0.6 and clients["qwen36"].profile.top_k == 20


def test_vertex_request_and_token_refresh(tmp_path, monkeypatch):
    from stirrup.core.models import SystemMessage, UserMessage

    monkeypatch.setenv("VERTEX_PROJECT", "test-project")
    client = providers.make_client(read_json(CONFIGS / "gdpval_sonnet46.json"), tmp_path / "calls.jsonl")
    tokens = iter(["expired", "fresh"])
    monkeypatch.setattr(providers.VertexToken, "get", lambda self: next(tokens))
    sent, sleeps = [], []
    replies = [(401, {"error": "UNAUTHENTICATED"}),
               (200, {"content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn"})]

    class FakeAsyncClient:
        def __init__(self, timeout):
            assert timeout == 1800

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, url, headers, json):
            sent.append((url, headers["Authorization"], json))
            status, body = replies.pop(0)
            return providers.httpx.Response(status, json=body)

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(providers.httpx, "AsyncClient", FakeAsyncClient)
    monkeypatch.setattr(providers.asyncio, "sleep", fake_sleep)
    message = asyncio.run(client.generate([SystemMessage(content="sys"), UserMessage(content="hi")], {}))
    assert message.content == "ok" and sleeps == [1.0]
    assert [auth for _, auth, _ in sent] == ["Bearer expired", "Bearer fresh"]
    url, _, payload = sent[0]
    assert url.endswith("/projects/test-project/locations/global/publishers/anthropic/models/claude-sonnet-4-6:rawPredict")
    assert payload == {"anthropic_version": "vertex-2023-10-16", "max_tokens": 65536, "temperature": 1.0,
                       "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}], "system": "sys",
                       "thinking": {"type": "adaptive", "display": "summarized"}, "output_config": {"effort": "max"}}


# ---- JobBench --------------------------------------------------------------

def test_content_cap():
    small = "=== FILE: a.txt ===\nshort\n"
    assert jobbench.cap_contents(small, 150_000) == small
    big = "=== FILE: a.txt ===\n" + "a" * 1000 + "\n" + "=== FILE: b.txt ===\n" + "b" * 50 + "\n"
    capped = jobbench.cap_contents(big, 400)
    assert len(capped) <= 400
    assert capped.endswith("=== FILE: b.txt ===\n" + "b" * 50 + "\n")        # a file within its share stays whole
    assert capped.startswith("=== FILE: a.txt ===\naaa")
    assert "[Gemini evidence truncated for context; middle omitted]" in capped


def _fake_jobbench_upstream(contents="=== FILE: out.md ===\nanswer\n", exit_codes=(0, 0)):
    def failed(index, rubric, reason):
        return {"index": index, "weight": rubric["weight"], "result": {"passed": False, "score": 0}, "reason": reason}

    def judge_rubric(index, rubric, file_contents, model, api_base, api_key, timeout, retries, images):
        passed = rubric["rubric"] == "pass"
        result = {"index": index, "weight": rubric["weight"],
                  "result": {"passed": passed, "score": rubric["weight"] if passed else 0}}
        return result, {"api_exit_code": exit_codes[index]}

    return SimpleNamespace(build_failed_rubric_result=failed, judge_rubric=judge_rubric,
                           extract_all_file_contents=lambda path: contents,
                           collect_image_attachments=lambda path: [])


RUBRICS = [{"rubric": "pass", "weight": 3}, {"rubric": "fail", "weight": 1}]


def test_jobbench_status_rules(tmp_path):
    settings = read_json(CONFIGS / "jobbench_gemini35flash.json")
    submission = tmp_path / "submission"
    submission.mkdir()
    upstream = _fake_jobbench_upstream()

    def status(record_status, upstream=upstream):
        return jobbench.judge_task(upstream, RUBRICS, submission, record_status, settings)[0]

    assert status("infra_error") == "infra_error"                     # unavailable
    assert status("abandoned") == "model_failure"                     # model zero
    assert status("complete") == "model_failure"                      # no files
    (submission / "empty.md").write_text("")
    assert status("complete") == "model_failure"                      # only empty files
    (submission / "out.md").write_text("answer")
    state, results = jobbench.judge_task(upstream, RUBRICS, submission, "complete", settings)
    assert state == "ok" and [row["result"]["passed"] for row in results] == [True, False]
    missing = _fake_jobbench_upstream("=== FILE: a.docx ===\n[ERROR: mammoth not available for a.docx]\n")
    assert status("complete", missing) == "infra_error"
    assert status("complete", _fake_jobbench_upstream("   ")) == "model_failure"
    assert status("complete", _fake_jobbench_upstream(exit_codes=(0, 2))) == "infra_error"


def test_jobbench_aggregate():
    rows = ([{"split": "main", "eligible": True, "normalized_score": 0.2}] * 65
            + [{"split": "easy", "eligible": True, "normalized_score": 0.6}] * 63)
    summary = jobbench.aggregate(rows)
    assert summary["macro"] == round((65 * 0.2 + 63 * 0.6) / 128, 6)
    assert summary["by_split"] == {"main": 0.2, "easy": 0.6}
    rows = [{"split": "main", "eligible": True, "normalized_score": 0.0},           # model zero counts
            {"split": "main", "eligible": False, "normalized_score": None},         # unavailable does not
            {"split": "easy", "eligible": True, "normalized_score": 0.5}]
    assert jobbench.aggregate(rows) == {"tasks": 3, "eligible": 2, "unavailable": 1, "macro": 0.25,
                                        "by_split": {"main": 0.0, "easy": 0.5}}


def test_jobbench_transport(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    transport = jobbench.GeminiTransport(read_json(CONFIGS / "jobbench_gemini35flash.json"))
    transport.log_path = tmp_path / "calls.jsonl"
    prompt = "Judge.\n- criteria_results array must have exactly 2 items (one for each criterion)\n"
    messages = [{"role": "system", "content": "You are an evaluation judge."},
                {"role": "user", "content": [{"type": "text", "text": prompt},
                                             {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}]
    sent = []
    truncated = '{"criteria_results": [{"index": 0, "passed": true, "reasoning": "r", "evidence": "e"}], ' \
                '"rubric_passed": false, "overall_reasoning": "o"'

    def fake_urlopen(request, timeout):
        sent.append((request, timeout))
        body = {"candidates": [{"content": {"parts": [{"text": "thinking", "thought": True}, {"text": truncated}]},
                                "finishReason": "MAX_TOKENS"}]}
        return io.BytesIO(json.dumps(body).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    client = transport.client("unused", "unused")
    reply = client.chat.completions.create(model="m", messages=messages, max_completion_tokens=8192,
                                           temperature=0.0, timeout=300)
    assert json.loads(reply.choices[0].message.content)["rubric_passed"] is False   # closed after truncation
    request, timeout = sent[0]
    payload = json.loads(request.data)
    assert timeout == 300 and request.headers["X-goog-api-key"] == "test-key"
    assert payload["systemInstruction"]["parts"][0]["text"] == \
        "You are an evaluation judge.\n\n" + jobbench.JSON_FORMAT_GUARD
    config = payload["generationConfig"]
    assert (config["maxOutputTokens"], config["temperature"], config["thinkingConfig"]) == \
        (65536, 0.0, {"thinkingLevel": "HIGH"})
    assert config["responseSchema"]["properties"]["criteria_results"]["minItems"] == 2
    assert payload["contents"][0]["parts"][1] == {"inlineData": {"mimeType": "image/png", "data": "AAAA"}}


# ---- LAB -------------------------------------------------------------------

def test_lab_parse_verdict():
    assert lab.parse_verdict('{"verdict": "PASS", "reasoning": "ok"}') == {"verdict": "pass", "reasoning": "ok"}
    assert lab.parse_verdict('{"verdict": "fail", "reasoning": "cut off')["verdict"] == "fail"
    with pytest.raises(ValueError):
        lab.parse_verdict('{"verdict": "maybe"}')
    with pytest.raises(ValueError):
        lab.parse_verdict("no json")


def test_lab_extraction_guard():
    scoring = SimpleNamespace(_read_file_as_text=lambda path, **kwargs: kwargs["text"])
    lab.guard(scoring)
    assert scoring._read_file_as_text(Path("a.md"), text="body") == "body"
    native = "(error reading a.docx: pandoc failed: couldn't parse docx file)"
    assert scoring._read_file_as_text(Path("a.docx"), text=native) == native     # upstream's model failure
    with pytest.raises(lab.ArtifactExtractionError):
        scoring._read_file_as_text(Path("a.pdf"), text="(error reading a.pdf: boom)")
    assert scoring._llm_match_deliverables({"x": "x.docx"}, ["y.docx"], Path(".")) == {}


def _lab_fixture(tmp_path, status):
    harvey = tmp_path / "harvey-labs"
    task_dir = harvey / "tasks" / "area" / "t1"
    task_dir.mkdir(parents=True)
    criteria = [{"id": f"C-{index}", "title": f"c{index}", "match_criteria": "m"} for index in range(3)]
    (task_dir / "task.json").write_text(json.dumps({"title": "T", "criteria": criteria}))
    run = tmp_path / "run"
    if status is not None:
        (run / "area" / "t1" / "final" / "submission").mkdir(parents=True)
        (run / "area" / "t1" / "final" / "submission" / "memo.docx").write_text("x")
        (run / "area" / "t1" / "result.json").write_text(json.dumps({"status": status}))
    return harvey, run


def test_lab_status_rules(tmp_path):
    profile = read_json(CONFIGS / "lab_gemini31pro.json")
    harvey, run = _lab_fixture(tmp_path, "abandoned")
    row = lab.score_task(None, profile, harvey, "area/t1", run, tmp_path / "out")
    assert (row["status"], row["eligible"], row["n_passed"], row["n_criteria"]) == ("incomplete", False, 0, 3)
    harvey, run = _lab_fixture(tmp_path / "missing", None)
    assert lab.score_task(None, profile, harvey, "area/t1", run, tmp_path / "out")["status"] == "unavailable"

    harvey, run = _lab_fixture(tmp_path / "ok", "complete")
    seen = {}

    def score_rubric(criteria, run_dir, judge, task_desc, parallel):
        seen.update(files=sorted(path.name for path in (run_dir / "output").iterdir()), parallel=parallel)
        verdicts = ["pass", "pass", "fail"]
        return SimpleNamespace(score=0.0, max_score=1.0, criteria_results=[
            {"id": c["id"], "title": c["title"], "verdict": v, "reasoning": ""} for c, v in zip(criteria, verdicts)])

    row = lab.score_task(SimpleNamespace(score_rubric=score_rubric), profile, harvey, "area/t1", run, tmp_path / "out")
    assert (row["status"], row["eligible"], row["n_passed"], row["all_pass"]) == ("ok", True, 2, False)
    assert seen == {"files": ["memo.docx"], "parallel": 4}

    def broken(**kwargs):
        raise RuntimeError("Judge failed after 3 attempts")

    row = lab.score_task(SimpleNamespace(score_rubric=broken), profile, harvey, "area/t1", run, tmp_path / "out")
    assert (row["status"], row["eligible"]) == ("unavailable", False)


def test_lab_judge_request(tmp_path, monkeypatch):
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    (prompts / "rubric_criterion.txt").write_text("Task {task_description}\n```json\n{{\"verdict\": ...}}\n```")
    judge = lab.GeminiJudge(read_json(CONFIGS / "lab_gemini31pro.json"), prompts, tmp_path / "calls.jsonl")
    sent = []
    replies = [RuntimeError("gemini_generate_content failed"),
               {"candidates": [{"content": {"parts": [{"text": "x", "thought": True},
                                                      {"text": '{"verdict": "pass", "reasoning": "r"}'}]}}]}]

    def fake_generate(model, payload, *, timeout, log_path):
        assert (model, timeout) == ("gemini-3.1-pro-preview", 1800)
        sent.append(payload)
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(lab.llm, "gemini_generate", fake_generate)
    assert judge.evaluate_from_file("rubric_criterion", {"task_description": "T"}) == {"verdict": "pass", "reasoning": "r"}
    assert sent[0]["contents"][0]["parts"][0]["text"] == 'Task T\n```json\n{"verdict": ...}\n```'
    assert sent[0]["systemInstruction"]["parts"][0]["text"] == lab.JUDGE_SYSTEM_PROMPT
    assert sent[0]["generationConfig"] == {"maxOutputTokens": 16384, "temperature": 0.0,
                                           "thinkingConfig": {"thinkingLevel": "high"},
                                           "responseMimeType": "application/json",
                                           "responseSchema": lab.VERDICT_SCHEMA}


def test_lab_aggregate():
    rows = [{"status": "ok", "eligible": True, "n_passed": 3, "n_criteria": 4, "all_pass": False},
            {"status": "ok", "eligible": True, "n_passed": 1, "n_criteria": 1, "all_pass": True},
            {"status": "incomplete", "eligible": False, "n_passed": 0, "n_criteria": 5, "all_pass": False},
            {"status": "unavailable", "eligible": False, "n_criteria": 2}]
    summary = lab.aggregate(rows)
    assert summary["macro"] == pytest.approx((0.75 + 1.0) / 2)
    assert summary["pooled"] == pytest.approx(4 / 5)
    assert (summary["eligible"], summary["incomplete"], summary["unavailable"], summary["all_pass"]) == (2, 1, 1, 1)
