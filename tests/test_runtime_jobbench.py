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
"""JobBench runtime with a scripted run_agent and a two-task fake dataset: staging and
evidence, prompt assembly, Loop-In phases, statuses, tool guards and submission. No API calls.

    python3 -m pytest -q tests/test_runtime_jobbench.py
"""

import asyncio
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from stirrup.core.agent import _SESSION_STATE                                # noqa: E402
from stirrup.tools.code_backends.base import CodeExecutionParams             # noqa: E402

from reorg.reflection_skill import chronicle as chron  # noqa: E402
from reorg import prompts  # noqa: E402
from reorg.domain import DOMAINS, load_domain, load_module  # noqa: E402
from reorg.files import read_json, write_jsonl  # noqa: E402
from reorg.llm import ModelProfile  # noqa: E402
from reorg.reflection_skill.reflect import Action, Contribution, reflect, run_action, work  # noqa: E402
from reorg.reflection_skill.graph import INPUT, OUTPUT, Node, ReflectionSkill  # noqa: E402
from reorg.worker import AgentRun  # noqa: E402

runtime = load_module(DOMAINS / "jobbench" / "runtime.py")
tools = runtime.tools
PROFILE = ModelProfile(name="test", provider="gemini", model="gemini-test", images=True)
MAIN, EASY = "main/clerks__task1", "easy/clerks__task1"


def _data(root: Path) -> Path:
    """<root>/jobbench with one main and one easy task, as prepare_data lays them out."""
    data = root / "jobbench"
    rows = []
    for task_id in (MAIN, EASY):
        split = task_id.split("/")[0]
        ref = Path(split) / "clerks" / "task1" / "task_folder" / "ledger.csv"
        (data / ref).parent.mkdir(parents=True)
        (data / ref).write_text("id,amount\n1,10\n", encoding="utf-8")
        rows.append({"canonical_task_id": task_id, "split": split, "prompt": f"Reconcile the ledger ({split}).",
                     "visible_reference_files": [ref.as_posix()]})
    write_jsonl(data / "tasks.jsonl", rows)
    return root


class Model:
    """Stands in for run_agent: records each session, then lets `act` work in its sandbox."""

    def __init__(self, act=None):
        self.calls, self.act = [], act or self.default

    def __call__(self, profile, *, system_prompt, prompt, tools, finish_tools, logs, name, max_turns, **limits):
        guard = tools[0]
        call = SimpleNamespace(name=name, system=system_prompt, prompt=prompt, sandbox=guard._sandbox,
                               outputs=guard.outputs, guard=guard, max_turns=max_turns, finish_tools=finish_tools,
                               tools=[getattr(tool, "name", type(tool).__name__) for tool in tools],
                               chronicle=(guard._sandbox / chron.STAGED).read_text(encoding="utf-8"),
                               artifact=sorted(p.name for p in (guard._sandbox / "current_artifact").iterdir()),
                               evidence=sorted(p.relative_to(guard._sandbox / "evidence").as_posix()
                                               for p in (guard._sandbox / "evidence").rglob("*") if p.is_file()))
        self.calls.append(call)
        finish, finish_type = self.act(call)
        return AgentRun(finish=finish, finish_type=finish_type, history=[], usage={"input": 2, "answer": 1, "reasoning": 0})

    @staticmethod
    def default(call):
        box = call.sandbox
        if call.outputs is None:
            (box / "current_artifact" / "log.xlsx").write_text(f"{call.name}\n")
            (box / "evidence" / f"{call.name}.pdf").write_text("source")
            return {"reason": "done", "paths": ["current_artifact/log.xlsx"]}, "JobBenchFinishParams"
        (box / "evidence" / f"{call.name}.txt").write_text("notes")
        (box / call.outputs[0]).write_text(f"# Handoff Summary\n\n{call.name} advice\n")
        return {"reason": "done", "paths": call.outputs}, "JobBenchFinishParams"


def _runtime(tmp: Path, task_id: str = MAIN):
    domain = load_domain("jobbench", _data(tmp))
    return runtime.make_runtime(domain, task_id, PROFILE, {}), domain


# ---- staging and evidence ------------------------------------------------------

def test_stage_and_evidence_flow_through_rounds():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        rt, domain = _runtime(tmp)
        model = Model()
        chronicle = chron.WorkChronicle(tmp / "run" / "work_chronicle.md")
        skill = ReflectionSkill("s", (Node("a", "direct_manager", "feedback"), Node("b", "internal_auditor", "propose")),
                                ((INPUT, "a"), ("a", "b"), ("b", OUTPUT))).validate()
        with mock.patch.object(runtime, "run_agent", model):
            first = work(Action("worker_initial", "initial_work", None), rt, None, chronicle, tmp / "run" / "a0")
            result = reflect(skill, rt, first.artifact, chronicle, tmp / "run" / "round_1")
        initial, a, b, revise = model.calls
        box = initial.sandbox
        assert (box / "task.md").read_text(encoding="utf-8") == "Reconcile the ledger (main).\n"
        link = box / "reference_files" / "main" / "clerks" / "task1" / "task_folder" / "ledger.csv"
        assert link.is_symlink() and link.resolve() == (domain.data / "main/clerks/task1/task_folder/ledger.csv").resolve()
        assert initial.artifact == [] and (box / "scratch").is_dir()
        assert (tmp / "run/a0/nodes/worker_initial/attempt_01/evidence/jobbench_worker_initial.pdf").is_file()
        # Round 1 sees the artifact and the initial work's evidence; b also sees a's evidence.
        assert initial.evidence == [] and a.artifact == ["log.xlsx"]
        assert a.evidence == ["a0/worker_initial/jobbench_worker_initial.pdf"]
        assert b.evidence == ["a0/worker_initial/jobbench_worker_initial.pdf", "upstream/a/jobbench_a.txt"]
        assert revise.evidence == ["a0/worker_initial/jobbench_worker_initial.pdf", "upstream/a/jobbench_a.txt",
                                   "upstream/b/jobbench_b.txt"]
        # Only what an action added is kept, not what it was given.
        assert sorted(p.name for p in (tmp / "run/round_1/nodes/b/attempt_01/evidence").rglob("*")) == ["jobbench_b.txt"]
        assert result.status == "strict" and (result.artifact / "log.xlsx").read_text() == "jobbench_worker_revise\n"


# ---- prompts, tools, statuses --------------------------------------------------

def test_wrappers_match_manifest():
    manifest = read_json(DOMAINS / "jobbench" / "prompts" / "manifest.json")["files"]
    assert set(manifest) == {"initial_work.md", "revise.md", "feedback.md", "propose.md", "loop_in.md"}
    for name, digest in manifest.items():
        assert hashlib.sha256((DOMAINS / "jobbench" / "prompts" / name).read_bytes()).hexdigest() == digest


def test_action_prompt_assembly():
    incoming = (Contribution.create("x", "cross_functional_partner", "feedback", "# Handoff Summary\n\nA.", "available", ()),
                Contribution.create("y", "technical_customer", "loop_in", "# Handoff Summary\n\nmissing", "missing", ()))
    prompt = runtime.action_prompt(Action("worker_revise", "revise", None, incoming), "Do the task.\n")
    assert prompt == "\n\n".join([
        "# Role / Social Profile\n\n" + prompts.load("worker/worker_role.md").strip(),
        "# Frozen Source Action Instruction\n\n" + prompts.load("worker/worker_revision.md").strip(),
        "# JobBench Transfer Wrapper\n\n" + (DOMAINS / "jobbench/prompts/revise.md").read_text().strip(),
        "# Original Task\n\nDo the task.\n",
        "# Incoming Contributions\n\n## xfn_partner_v003 / feedback / available\n\n# Handoff Summary\n\nA.\n\n"
        "## customer_technical_v003 / loop_in / missing\n\n# Handoff Summary\n\nmissing",
        "Work only inside the visible workspace. Begin now."])
    alone = runtime.action_prompt(Action("n", "propose", "internal_auditor"), "Do the task.")
    assert alone.startswith("# Role / Social Profile\n\n" + prompts.load("roles/internal_auditor.md").strip())
    assert "# Incoming Contributions\n\nNo colleague contributions are available.\n\n" in alone
    # Byte locks on the assembled prompt and the system prompt.
    assert hashlib.sha256(prompt.encode()).hexdigest() == (
        "398d4f69a6a86863ecba27542782952cece03635420042ef84c0acba8fdbfc45")
    assert hashlib.sha256(runtime.SYSTEM_PROMPT.encode()).hexdigest() == (
        "ac30d9dd9bf92da628ebc48823ec40c39cf5847d0f5224723f8f4372253133cb")


def test_tools_by_action_and_split():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        main, _ = _runtime(tmp / "m", MAIN)
        easy, _ = _runtime(tmp / "e", EASY)
        model = Model()
        chronicle = chron.WorkChronicle(tmp / "chronicle.md")
        with mock.patch.object(runtime, "run_agent", model):
            for rt, name in ((main, "m"), (easy, "e")):
                run_action(Action("worker_initial", "initial_work", None), rt, None, chronicle,
                           tmp / name / "run/a0/nodes/worker_initial", 0)
                run_action(Action("n", "feedback", "direct_manager"), rt, None, chronicle,
                           tmp / name / "run/round_1/nodes/n", 0)
        main_worker, main_feedback, easy_worker, easy_feedback = model.calls
    web = ["web_search", "fetch_web_page", "download_file"]
    assert main_worker.tools == ["GuardedCodeExec", *web, "ViewImageToolProvider"]
    assert easy_worker.tools == main_feedback.tools == easy_feedback.tools == ["GuardedCodeExec", "ViewImageToolProvider"]
    assert main_worker.system == runtime.SYSTEM_PROMPT and main_worker.max_turns == 250
    assert (main_worker.guard.outputs, main_worker.guard.post_output_limit) == (None, None)
    assert (main_feedback.outputs, main_feedback.guard.pre_output_limit, main_feedback.guard.post_output_limit) == (
        ["collaboration_workspace/colleague_feedback.md"], 50, 4)
    assert [tool.name for tool in main_worker.finish_tools] == ["finish", "abandon_task_finish"]


def test_statuses():
    def act(finish, finish_type, write=True):
        def inner(call):
            if write and call.outputs:
                (call.sandbox / call.outputs[0]).write_text("# Handoff Summary\n\nok\n")
            return finish, finish_type
        return inner

    cases = [(Action("worker_revise", "revise", None), act(None, None), "turn_limit_best_available"),
             (Action("worker_revise", "revise", None), act({"reason": "x"}, "AbandonTaskFinishParams"), "model_failure"),
             (Action("n", "propose", "direct_manager"), act(None, None), "model_failure"),
             (Action("n", "propose", "direct_manager"), act({"reason": "r", "paths": []}, "JobBenchFinishParams"), "ok")]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        rt, _ = _runtime(tmp)
        for index, (action, behaviour, status) in enumerate(cases):
            with mock.patch.object(runtime, "run_agent", Model(behaviour)):
                result, sandbox, _ = run_action(action, rt, None, chron.WorkChronicle(tmp / "c.md"),
                                                tmp / f"run/round_{index}/nodes/{action.vertex_id}", 0)
            assert result.status == status
            assert result.artifact == (sandbox / "current_artifact" if action.role is None else None)
        assert result.text == "# Handoff Summary\n\nok\n"


# ---- Loop-In -------------------------------------------------------------------

def test_loop_in_phases():
    def act(call):
        staged = call.sandbox / chron.STAGED
        staged.write_text(staged.read_text() + f"\n## Work Chronicle Update\n\n{call.name}\n")
        output = call.sandbox / call.outputs[0]
        if call.name == "jobbench_loop_in_select":
            output.write_text(json.dumps({"partner_display_name": "Internal auditor", "opening_message": "Agenda."}))
        elif output.suffix == ".json":
            output.write_text(json.dumps({"message": f"From {output.stem}."}))
        else:
            output.write_text("# Handoff Summary\n\nI consulted the internal auditor.\n")
        return {"reason": "r", "paths": call.outputs}, "JobBenchFinishParams"

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        rt, _ = _runtime(tmp)
        model = Model(act)
        chronicle = chron.WorkChronicle(tmp / "c.md")
        with mock.patch.object(runtime, "run_agent", model):
            result, sandbox, base = run_action(Action("li", "loop_in", "direct_manager"), rt, None, chronicle,
                                               tmp / "run/round_1/nodes/li", 0)
        merged = (sandbox / chron.STAGED).read_text()               # what reflect merges: the last phase's update
    assert [call.name for call in model.calls] == ["jobbench_loop_in_select", "jobbench_loop_in_partner_1",
                                                   "jobbench_loop_in_initiator_1", "jobbench_loop_in_publish"]
    assert [call.outputs[0].split("/")[-1] for call in model.calls] == [
        "selection.json", "partner_001.json", "initiator_001.json", "summary.md"]
    assert all(call.chronicle == base and call.max_turns == 250 for call in model.calls)   # restaged per phase
    assert all(call.tools == ["GuardedCodeExec", "ViewImageToolProvider"] for call in model.calls)
    partner = model.calls[1].prompt
    assert partner.startswith("# Role / Social Profile\n\n" + prompts.load("roles/internal_auditor.md").strip())
    assert "\n\n# Frozen Source Action\n\n" + prompts.load("loop_in/loop_in.md").strip() in partner
    assert "Message 1 - Direct manager:\nAgenda." in partner
    assert partner.endswith("# Phase Completion Adapter\n\nWrite the `send_message` content as JSON key `message` to "
                            "`collaboration_workspace/loop_in/partner_001.json`, then finish.")
    assert hashlib.sha256(model.calls[0].prompt.encode()).hexdigest() == (
        "d69e11e43d5ee8b6937f534054ecf05f73341528db0108d8f206bbb55e06f375")
    assert result.status == "ok" and result.usage == {"input": 8, "answer": 4, "reasoning": 0}
    assert result.text.startswith("# Handoff Summary\n\nI consulted the internal auditor.\n\n# Discussion Transcript")
    assert "## Round 1 - Internal auditor\n\nFrom partner_001." in result.text
    assert merged == base + "\n## Work Chronicle Update\n\njobbench_loop_in_publish\n"


def test_loop_in_phase_without_its_file_is_a_protocol_failure():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        rt, _ = _runtime(tmp)
        silent = Model(lambda call: ({"reason": "r", "paths": []}, "JobBenchFinishParams"))
        with mock.patch.object(runtime, "run_agent", silent):
            result, _, _ = run_action(Action("li", "loop_in", "direct_manager"), rt, None,
                                      chron.WorkChronicle(tmp / "c.md"), tmp / "run/round_1/nodes/li", 0)
    assert result.status == "protocol_failure" and result.logs["phase"] == "select" and len(silent.calls) == 1


# ---- guards --------------------------------------------------------------------

def test_retrieval_guards():
    assert tools.query_block_reason("main/clerks__task1 answer", task_id=MAIN, task_text="x")
    assert tools.query_block_reason("site:hf.co ledger", task_id=MAIN, task_text="x")
    assert tools.query_block_reason('"reconcile the ledger quickly"', task_id=MAIN,
                                    task_text="Reconcile the ledger quickly, then report.")
    assert tools.query_block_reason("2025 fee schedule", task_id=MAIN, task_text="x") is None
    blocked = ["ftp://cms.gov/a", "http://localhost/a", "http://192.168.0.1/a", "https://huggingface.co/x",
               "https://cdn.hf.co/x", "https://example.com/JobBench/x", f"https://example.com/{MAIN}"]
    assert all(tools.url_block_reason(url, task_id=MAIN) for url in blocked)
    assert tools.url_block_reason("https://www.cms.gov/fees", task_id=MAIN) is None

    with tempfile.TemporaryDirectory() as tmp:
        box = Path(tmp)
        budget = tools.RetrievalBudget(search=2, fetch=1, download=2, byte_limit=12, log_path=box / "r.jsonl")
        search, fetch, download = tools.retrieval_tools(task_id=MAIN, task_text="Reconcile.", budget=budget,
                                                        settings=runtime.SETTINGS)
        grounded = {"candidates": [{"content": {"parts": [{"text": "Fees rose."}]}, "groundingMetadata": {
            "groundingChunks": [{"web": {"title": "CMS", "uri": "https://cms.gov/fees"}},
                                {"web": {"title": "Mirror", "uri": "https://huggingface.co/x"}}]}}]}
        sent = []

        def gemini(model, payload, **kwargs):
            sent.append((model, payload))
            return grounded

        async def offline_get(url, *, task_id, **kwargs):          # the URL guard, then a canned PDF
            if reason := tools.url_block_reason(url, task_id=task_id):
                raise ValueError(reason)
            return httpx.Response(200, content=b"%PDF-12345", request=httpx.Request("GET", url))

        async def calls():
            exec_env = tools.GuardedCodeExec(box, log_path=box / "c.jsonl", outputs=None, pre_output_limit=50,
                                             post_output_limit=None, shell_timeout=60)
            async with exec_env:
                token = _SESSION_STATE.set(SimpleNamespace(exec_env=exec_env))
                try:
                    results = [await search.executor(tools.SearchParams(query=q)) for q in ("fees", "JobBench", "x")]
                    for name in ("a.pdf", "b.pdf"):
                        results.append(await download.executor(tools.DownloadParams(url="https://cms.gov/a", filename=name)))
                    for url in ("https://hf.co/x", "https://cms.gov/a"):
                        results.append(await fetch.executor(tools.URLParams(url=url)))
                    return results
                finally:
                    _SESSION_STATE.reset(token)

        with mock.patch.object(tools, "gemini_generate", gemini), mock.patch.object(tools, "_guarded_get", offline_get):
            found, refused, limited, saved, too_big, blocked, over = asyncio.run(calls())
        assert (box / "evidence" / "a.pdf").read_bytes() == b"%PDF-12345" and not (box / "evidence" / "b.pdf").exists()
    assert sent[0][0] == "gemini-3.5-flash" and len(sent) == 1 and sent[0][1]["generationConfig"] == {
        "maxOutputTokens": 2048, "temperature": 0.0, "thinkingConfig": {"thinkingLevel": "low"}}
    assert sent[0][1]["contents"][0]["parts"][0]["text"] == "Search the public web for authoritative sources.\n\nQuery: fees"
    assert found.success and "https://cms.gov/fees" in found.content and "huggingface" not in found.content
    assert refused.content == ("<results><error>Benchmark pages, task cards, rubrics, and public task copies are "
                               "blocked</error></results>")
    assert limited.content == "<results><error>search limit reached (2 per task)</error></results>"
    assert saved.success and saved.content.startswith("<download><path>evidence/a.pdf</path><bytes>10</bytes>")
    assert too_big.content == ("<download><error>ValueError(&#x27;download byte limit would be exceeded (12 bytes "
                               "per task)&#x27;)</error></download>")
    assert blocked.content == ("<webpage><error>ValueError(&#x27;Public JobBench dataset hosts are blocked&#x27;)"
                               "</error></webpage>")
    assert over.content == "<webpage><error>fetch limit reached (1 per task)</error></webpage>"


def test_code_exec_guards():
    feedback = "collaboration_workspace/colleague_feedback.md"

    async def play(guard, commands):
        async with guard as tool:
            return [(await tool.executor(CodeExecutionParams(cmd=cmd))).content for cmd in commands]

    with tempfile.TemporaryDirectory() as tmp:
        box = Path(tmp) / "sandbox"
        (box / "collaboration_workspace").mkdir(parents=True)
        guard = tools.GuardedCodeExec(box, log_path=Path(tmp) / "c.jsonl", outputs=[feedback], pre_output_limit=3,
                                      post_output_limit=1, shell_timeout=60)
        out = asyncio.run(play(guard, ["curl https://example.com", "echo a", "echo b", "echo c",
                                       f"echo '# Handoff Summary' > {feedback}", "echo d", "echo e"]))
    assert "Direct network retrieval is disabled" in out[0]
    assert "Complete the review now" in out[1] and "Complete the review now" in out[2]
    assert "The analysis call budget is exhausted" in out[3]
    assert "You have at most 1 more code_exec calls" in out[4] and "at most 0 more" in out[5]
    assert "post-output tool budget is exhausted" in out[6]


# ---- submission ----------------------------------------------------------------

def test_submission_is_the_final_artifact():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        artifact = tmp / "final_artifact"
        (artifact / "tables").mkdir(parents=True)
        (artifact / "error_log.xlsx").write_bytes(b"xlsx")
        (artifact / "tables" / "review.csv").write_text("a,b\n")
        out = tmp / "final" / "submission"
        (out).mkdir(parents=True)
        (out / "stale.txt").write_text("old")
        files = runtime.submission(None, MAIN, artifact, out)
        assert files == ["error_log.xlsx", "tables/review.csv"]
        assert sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()) == files
