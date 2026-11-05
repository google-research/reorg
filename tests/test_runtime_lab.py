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
"""LAB runtime with a stubbed agent on a fake harvey-labs checkout: staging, prompt
assembly, tools, statuses, the Loop-In driver, the finalization guard and submission.
No API calls."""

import asyncio
import hashlib
import json
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stirrup.core.agent import _SESSION_STATE  # noqa: E402
from stirrup.tools import ViewImageToolProvider  # noqa: E402
from stirrup.tools.code_backends.base import CommandResult  # noqa: E402

from reorg.reflection_skill import chronicle as chron  # noqa: E402
from reorg.reflection_skill import loop_in
from reorg import prompts  # noqa: E402
from reorg.domain import DOMAINS, load_domain, load_module  # noqa: E402
from reorg.files import read_json, write_jsonl  # noqa: E402
from reorg.llm import load_profile  # noqa: E402
from reorg.reflection_skill.reflect import Action, Contribution  # noqa: E402
from reorg.worker import AgentRun, WorkspaceCodeExec  # noqa: E402

rt = load_module(DOMAINS / "lab" / "runtime.py")
prepare_data = load_module(DOMAINS / "lab" / "prepare_data.py")
TASK_ID = "real-estate/lease-review/scenario-01"
REQUEST = "Review the lease in `documents/` and produce:\n1. `lease-memo.docx`\n2. `rent-schedule.xlsx`\n"
MEMO, SCHEDULE = "current_artifact/lease-memo.docx", "current_artifact/rent-schedule.xlsx"
FINISH_TOOL = rt.tools.finish_tool
EXPECTED: dict[int, list[str]] = {}                  # id(finish tool) -> the outputs it requires


def _domain(tmp: Path):
    task_dir = tmp / "lab" / "harvey-labs" / "tasks" / TASK_ID
    (task_dir / "documents" / "exhibits").mkdir(parents=True)
    (task_dir / "task.json").write_text(json.dumps({
        "title": "Lease review", "instructions": REQUEST, "work_type": "review",
        "deliverables": {"memo": "lease-memo.docx", "schedule": "rent-schedule.xlsx"},
        "criteria": [{"id": "C-001", "title": "Term", "match_criteria": "States the term."}]}))
    (task_dir / "documents" / "lease.txt").write_text("Term: 5 years.\n")
    (task_dir / "documents" / "exhibits" / "rent_roll.csv").write_text("unit,rent\n")
    write_jsonl(tmp / "lab" / "tasks.jsonl", prepare_data.discover_tasks(tmp / "lab" / "harvey-labs"))
    return load_domain("lab", tmp)


def _runtime(tmp: Path):
    domain = _domain(tmp)
    return domain, rt.make_runtime(domain, TASK_ID, load_profile("gemini_3_1_pro"), {})


def _office(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("[Content_Types].xml", path.name)


class FakeAgent:
    """Stands in for run_agent: records each session and plays a scripted model."""

    def __init__(self, write=True, finish=True, partner="Legal and risk reviewer"):
        self.calls, self.write, self.finish, self.partner = [], write, finish, partner

    def __call__(self, profile, **kwargs):
        sandbox, expected = kwargs["tools"][0]._sandbox, EXPECTED[id(kwargs["finish_tools"][0])]
        self.calls.append({**kwargs, "expected": expected,
                           "chronicle": (sandbox / chron.STAGED).read_text()})
        for relative in expected if self.write else ():
            path = sandbox / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            if relative.startswith("current_artifact/") and self.write == "invalid":
                path.write_text("not a zip")
            elif relative.startswith("current_artifact/"):
                _office(path)
            elif relative.endswith("selection.json"):
                path.write_text(json.dumps({"partner_display_name": self.partner, "opening_message": "Agenda."}))
            elif relative.endswith(".json"):
                path.write_text(json.dumps({"message": f"From {path.stem}."}))
            else:
                path.write_text(f"# Handoff Summary\n\nFrom {path.name}.\n")
        with (sandbox / chron.STAGED).open("a") as handle:
            handle.write("\n## Work Chronicle Update\n\nAppended.\n")
        finish = {"reason": "done", "paths": expected} if self.finish else None
        return AgentRun(finish, "LabFinishParams" if self.finish else None, [], {"input": 2, "answer": 1})


def _finish_tool(expected):
    tool = FINISH_TOOL(expected)
    EXPECTED[id(tool)] = list(expected)
    return tool


def _run(runtime, action, box, agent, artifact=None, chronicle=chron.INITIAL):
    runtime.stage(box, artifact)
    (box / chron.STAGED).parent.mkdir(parents=True)
    (box / chron.STAGED).write_text(chronicle)
    with mock.patch.object(rt, "run_agent", agent), mock.patch.object(rt.tools, "finish_tool", _finish_tool):
        return runtime.run(action, box, box.parent / "logs")


def test_wrappers_match_manifest():
    manifest = read_json(DOMAINS / "lab" / "prompts" / "manifest.json")["files"]
    assert set(manifest) == {f"{action}.md" for action in rt.SOURCE_ACTIONS}
    for name, entry in manifest.items():
        assert hashlib.sha256((DOMAINS / "lab" / "prompts" / name).read_bytes()).hexdigest() == entry["sha256"]


def test_stage_layout(tmp_path):
    domain, runtime = _runtime(tmp_path)
    box = tmp_path / "box"
    runtime.stage(box, None)
    assert (box / "task.md").read_text() == REQUEST + "\n"
    documents = box / "reference_files" / "tasks" / TASK_ID / "documents"
    assert (documents / "lease.txt").is_symlink()
    source = domain.harvey_root / "tasks" / TASK_ID / "documents"
    assert (documents / "lease.txt").resolve() == (source / "lease.txt").resolve()
    assert (documents / "exhibits" / "rent_roll.csv").read_text() == "unit,rent\n"
    assert list((box / "current_artifact").iterdir()) == [] and (box / "scratch").is_dir()
    assert not (box / "skills").exists()                     # no document-skill manuals

    artifact = tmp_path / "a0"
    _office(artifact / "lease-memo.docx")
    runtime.stage(tmp_path / "box2", artifact)
    copied = tmp_path / "box2" / "current_artifact" / "lease-memo.docx"
    assert copied.read_bytes() == (artifact / "lease-memo.docx").read_bytes()


def test_worker_initial_work_and_revision(tmp_path):
    _, runtime = _runtime(tmp_path)
    agent = FakeAgent()
    result = _run(runtime, Action("worker_initial", "initial_work", None), tmp_path / "a", agent)
    assert result.status == "ok" and result.artifact == tmp_path / "a" / "current_artifact"
    call = agent.calls[0]
    assert call["system_prompt"] == rt.LAB_SYSTEM_PROMPT + "\n\n" + rt.FINALIZATION_STEERING_PROMPT
    assert call["prompt"] == "\n\n".join([
        "# Role / Social Profile\n\n" + prompts.load("worker/worker_role.md").strip(),
        "# Frozen Source Action Instruction\n\n" + prompts.load("worker/initial_work.md").strip(),
        "# LAB Transfer Wrapper\n\n" + (DOMAINS / "lab" / "prompts" / "initial_work.md").read_text().strip(),
        "# Original Task\n\n" + REQUEST,
        "# Incoming Contributions\n\nNo colleague contributions are available.",
        "Use trusted-local execution only. Begin now."])
    code, view = call["tools"]
    assert isinstance(code, rt.tools.FinalizationGuard) and isinstance(view, ViewImageToolProvider)
    assert (code.expected, code.pre_output_calls, code.post_output_calls) == ([MEMO, SCHEDULE], 50, 4)
    assert [tool.name for tool in call["finish_tools"]] == ["finish", "abandon_task_finish"]
    assert call["max_turns"] == 250 and call["name"] == "lab_worker_initial"

    incoming = (Contribution.create("x", "cross_functional_partner", "feedback", "# Handoff Summary\n\nFix rent.",
                                    "available", ()),)
    agent = FakeAgent()
    result = _run(runtime, Action("worker_revise", "revise", None, incoming), tmp_path / "r", agent,
                  artifact=tmp_path / "a" / "current_artifact")
    call = agent.calls[0]
    assert result.status == "ok"
    assert call["system_prompt"] == rt.LAB_SYSTEM_PROMPT                       # revision is unbounded
    assert type(call["tools"][0]) is WorkspaceCodeExec
    # Contributions are headed by the model-facing role ID (xfn_partner_v003).
    assert ("# Incoming Contributions\n\n## xfn_partner_v003 / feedback / available\n\n# Handoff Summary\n\nFix rent."
            in call["prompt"])
    assert "# LAB Transfer Wrapper\n\n# LAB task-owner Improve" in call["prompt"]


def test_status_rules(tmp_path):
    _, runtime = _runtime(tmp_path)
    initial = Action("worker_initial", "initial_work", None)
    assert _run(runtime, initial, tmp_path / "t", FakeAgent(finish=False)).status == "turn_limit_best_available"
    assert _run(runtime, initial, tmp_path / "i", FakeAgent(write="invalid")).status == "protocol_failure"
    assert _run(runtime, initial, tmp_path / "n", FakeAgent(write=False)).status == "protocol_failure"

    feedback = Action("n1", "feedback", "direct_manager")
    salvaged = _run(runtime, feedback, tmp_path / "f", FakeAgent(finish=False))
    assert salvaged.status == "ok" and salvaged.text.startswith("# Handoff Summary")
    failed = _run(runtime, feedback, tmp_path / "g", FakeAgent(write=False, finish=False))
    assert (failed.status, failed.text) == ("model_failure", "")


def test_stakeholder_prompt(tmp_path):
    _, runtime = _runtime(tmp_path)
    agent = FakeAgent()
    _run(runtime, Action("n1", "propose", "legal_risk_reviewer"), tmp_path / "p", agent)
    call = agent.calls[0]
    assert call["expected"] == ["collaboration_workspace/improvement_proposal.md"]
    assert call["prompt"].startswith(
        "# Role / Social Profile\n\n" + prompts.load("roles/legal_risk_reviewer.md").strip()
        + "\n\n# Frozen Source Action Instruction\n\n" + prompts.load("actions/propose.md").strip())
    assert call["system_prompt"].endswith(rt.FINALIZATION_STEERING_PROMPT)


def test_loop_in_phases(tmp_path):
    _, runtime = _runtime(tmp_path)
    agent = FakeAgent()
    base = chron.INITIAL + "\n## Prior\n"
    result = _run(runtime, Action("n1", "loop_in", "direct_manager"), tmp_path / "l", agent, chronicle=base)
    assert result.status == "ok" and result.usage == {"input": 8, "answer": 4}
    assert [call["expected"] for call in agent.calls] == [
        ["collaboration_workspace/loop_in/selection.json"], ["collaboration_workspace/loop_in/partner_001.json"],
        ["collaboration_workspace/loop_in/initiator_001.json"], ["collaboration_workspace/loop_in/summary.md"]]
    roles = ["direct_manager", "legal_risk_reviewer", "direct_manager", "direct_manager"]
    for call, role in zip(agent.calls, roles):
        assert call["system_prompt"] == rt.LAB_SYSTEM_PROMPT + "\n\n" + prompts.load(f"roles/{role}.md").strip()
        assert call["prompt"].startswith("# Frozen Source Action\n\n" + prompts.load("loop_in/loop_in.md").strip())
        assert "\n\n# LAB Transfer Wrapper\n\n# LAB colleague Loop In" in call["prompt"]
        assert call["chronicle"] == base                       # every phase starts from the staged chronicle
        assert isinstance(call["tools"][0], rt.tools.FinalizationGuard) and call["max_turns"] == 250
    select = agent.calls[0]["prompt"]
    assert "- Same-job teammate\n- Upstream requester" in select and "- Direct manager" not in select
    assert select.endswith("# Phase Completion Adapter\n\nWrite JSON keys `partner_display_name` and `opening_message` "
                           "to `collaboration_workspace/loop_in/selection.json`, then finish with that path.")
    assert "Message 2 - Legal and risk reviewer:\nFrom partner_001." in agent.calls[2]["prompt"]
    assert result.text.startswith("# Handoff Summary\n\nFrom summary.md.\n\n# Discussion Transcript")
    assert (tmp_path / "l" / rt.LOOP_IN_OUTCOME).read_text() == result.text

    failed = _run(runtime, Action("n1", "loop_in", "direct_manager"), tmp_path / "m", FakeAgent(partner="Task Owner"))
    assert (failed.status, failed.text) == ("protocol_failure", "")


def test_finalization_guard(tmp_path):
    async def ran(self, cmd, *, timeout=None):
        return CommandResult(exit_code=0, stdout="ran", stderr="")

    guard = rt.tools.FinalizationGuard(tmp_path, [MEMO], pre_output_calls=2, post_output_calls=1, shell_timeout=5)
    with mock.patch.object(WorkspaceCodeExec, "run_command", ran):
        def call(cmd):
            return asyncio.run(guard.run_command(cmd))
        assert "Stop broad inspection" in call("ls").stdout
        assert call("ls").stdout.startswith("ran")
        blocked = call("ls")
        assert blocked.error_kind == "finish_required" and "analysis budget is exhausted" in blocked.stderr
        _office(tmp_path / MEMO)
        assert "Use at most 0 more code_exec calls" in call("python make_memo.py").stdout
        assert "validation budget is exhausted" in call("ls").stderr


def test_finish_tool(tmp_path):
    async def exists(path):
        return path == MEMO

    async def is_dir(path):
        return False

    tool = rt.tools.finish_tool([MEMO])
    token = _SESSION_STATE.set(SimpleNamespace(exec_env=SimpleNamespace(file_exists=exists, is_directory=is_dir)))
    try:
        wrong = asyncio.run(tool.executor(tool.parameters(reason="r", paths=["lease-memo.docx"])))
        right = asyncio.run(tool.executor(tool.parameters(reason="r", paths=[MEMO])))
    finally:
        _SESSION_STATE.reset(token)
    assert (wrong.success, wrong.content) == (False, f"ERROR: submit exactly these paths: {[MEMO]}")
    assert (right.success, right.content) == (True, "r")
    assert "description" not in tool.parameters.model_json_schema()      # no class docstring in the schema


def test_submission(tmp_path):
    domain, _ = _runtime(tmp_path)
    artifact = tmp_path / "artifact"
    _office(artifact / "lease-memo.docx")
    _office(artifact / "rent-schedule.xlsx")
    (artifact / "notes.md").write_text("draft")
    (artifact / ".hidden").write_text("x")
    out = tmp_path / "final" / "submission"
    assert rt.submission(domain, TASK_ID, artifact, out) == ["lease-memo.docx", "rent-schedule.xlsx"]
    assert sorted(path.name for path in out.iterdir()) == ["lease-memo.docx", "rent-schedule.xlsx"]
    manifest = read_json(tmp_path / "final" / "manifest.json")
    assert (manifest["missing_deliverables"], manifest["ignored_extra_files"]) == ([], ["notes.md"])

    (artifact / "rent-schedule.xlsx").unlink()
    assert rt.submission(domain, TASK_ID, artifact, tmp_path / "f2" / "submission") == []
    assert read_json(tmp_path / "f2" / "manifest.json")["missing_deliverables"] == ["rent-schedule.xlsx"]
