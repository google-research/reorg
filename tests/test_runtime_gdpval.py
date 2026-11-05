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
"""GDPval runtime with a stubbed agent: sandbox layout, prompt assembly per action,
the Loop-In driver and submission. No API calls."""

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stirrup.tools import ViewImageToolProvider  # noqa: E402

from reorg.reflection_skill import loop_in
from reorg import prompts  # noqa: E402
from reorg.domain import DOMAINS, TaskPacket, load_module  # noqa: E402
from reorg.llm import load_profile  # noqa: E402
from reorg.reflection_skill.reflect import Action, Contribution  # noqa: E402
from reorg.web_search import FetchTool, SearchTool  # noqa: E402
from reorg.worker import AgentRun, WorkspaceCodeExec  # noqa: E402

rt = load_module(DOMAINS / "gdpval" / "runtime.py")
adapter = load_module(DOMAINS / "gdpval" / "adapter.py")
GEMINI = load_profile("gemini_3_1_pro")
REQUEST = "Fill in the location report from the receiving log.\n"
REFERENCE = "reference_files/d70a/Blank Location Report.xlsx"
DELIVERABLE = "deliverable_files/feaf/Location Report Fix.xlsx"
OUTPUT = "collaboration_workspace/loop_in_outcome.md"


def _runtime(tmp: Path, references=(REFERENCE,)):
    data = tmp / "data"
    for path in references:
        (data / path).parent.mkdir(parents=True, exist_ok=True)
        (data / path).write_bytes(b"xlsx")
    packet = TaskPacket("t1", REQUEST, tuple(data / path for path in references), (DELIVERABLE,))
    domain = SimpleNamespace(data=data, load_task=lambda task_id: packet)
    return rt.make_runtime(domain, "t1", GEMINI, {}), data


class FakeAgent:
    """Stands in for run_agent: records each session, writes scripted files, returns a scripted finish."""

    def __init__(self, *script):
        self.calls, self.script = [], list(script)

    def __call__(self, profile, **kwargs):
        self.calls.append(kwargs)
        finish_type, finish, files = self.script.pop(0)
        for path, text in files.items():
            (kwargs["tools"][0]._sandbox / path).write_text(text)
        return AgentRun(finish, finish_type, [], {"input": 2, "answer": 1, "reasoning": 0})


def _run(runtime, action, box, *script, artifact=None):
    runtime.stage(box, artifact)
    (box / "collaboration_workspace").mkdir()
    agent = FakeAgent(*script)
    with mock.patch.object(rt, "run_agent", agent), \
            mock.patch.object(rt, "finish_tool", wraps=rt.finish_tool) as finish:
        result = runtime.run(action, box, box.parent / "logs")
    return result, agent.calls, [call.args[0] for call in finish.call_args_list]


def _finish(*paths):
    return "FinishParams", {"reason": "done", "paths": list(paths)}


def _contribution(vertex_id):
    return Contribution.create(vertex_id, "direct_manager", "feedback", "# Handoff Summary\n\nx", "available", ())


# ---- sandbox ---------------------------------------------------------------

def test_sandbox_layout():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        runtime, data = _runtime(tmp, (REFERENCE, "reference_files/63f4/Inv on line.xlsx"))
        artifact = tmp / "artifact"
        (artifact / "charts").mkdir(parents=True)
        (artifact / "Location Report Fix.xlsx").write_bytes(b"v1")
        (artifact / "charts" / "a.png").write_bytes(b"png")
        runtime.stage(tmp / "box", artifact)
        box = tmp / "box"
        assert sorted(p.relative_to(box).as_posix() for p in box.rglob("*")) == [
            "current_artifact", "current_artifact/Location Report Fix.xlsx", "current_artifact/charts",
            "current_artifact/charts/a.png", "reference_files", "reference_files/63f4",
            "reference_files/63f4/Inv on line.xlsx", "reference_files/d70a",
            "reference_files/d70a/Blank Location Report.xlsx", "scratch", "task.md"]
        assert (box / "task.md").read_bytes() == REQUEST.encode() + b"\n"
        link = box / "reference_files/d70a/Blank Location Report.xlsx"
        assert link.is_symlink() and Path(link.readlink()) == (data / REFERENCE).resolve()
        assert (box / "current_artifact/Location Report Fix.xlsx").read_bytes() == b"v1"
        runtime.stage(tmp / "first", None)
        assert not any((tmp / "first" / "current_artifact").iterdir()) and (tmp / "first" / "scratch").is_dir()


def test_adapter_declares_the_expert_deliverable_paths():
    domain = adapter.GDPval()
    domain.data = Path("/data/gdpval")
    domain._rows = {"t1": {"prompt": REQUEST, "reference_files": [REFERENCE], "deliverable_files": [DELIVERABLE]}}
    packet = domain.load_task("t1")
    assert packet.references == (Path("/data/gdpval") / REFERENCE,) and packet.deliverables == (DELIVERABLE,)
    runtime = rt.make_runtime(domain, "t1", GEMINI, {})
    assert runtime.references == {Path("d70a/Blank Location Report.xlsx"): Path("/data/gdpval") / REFERENCE}
    assert (runtime.settings["max_turns"], runtime.settings["loop_in_phase_max_turns"]) == (250, 30)


# ---- prompts per action ------------------------------------------------------

def _tail(count, finish, role, instruction, references=f"- `{REFERENCE}`"):
    return (f"Available colleague contribution count: {count}\n\n"
            f"Expected final deliverables declared for this task:\n\n"
            f"- `{DELIVERABLE}`\n\nFor this action, call `finish` with these relative path(s) when available:\n\n"
            f"{finish}\n\n## Role / Social Profile\n\n{prompts.load(role).strip()}\n\n## Action Instruction\n\n"
            f"{prompts.load(instruction).strip()}\n\n## Reference Files\n\n{references}\n\n## Original Task\n\n"
            f"{REQUEST}\n\nBegin the action now.\n")


def test_worker_prompts_tools_and_status():
    deliverable = "current_artifact/Location Report Fix.xlsx"
    with tempfile.TemporaryDirectory() as tmp:
        runtime, _ = _runtime(Path(tmp))
        result, calls, expected = _run(runtime, Action("worker_initial", "initial_work", None), Path(tmp) / "a0",
                                       (*_finish(deliverable), {deliverable: "x"}))
        call = calls[0]
        assert call["system_prompt"] == rt.ACTION_SYSTEM_PROMPT
        assert call["prompt"].startswith("## Project Workspace\n\nYou are working in a local Linux project workspace.")
        assert call["prompt"].endswith(_tail(0, f"- `{deliverable}`", "worker/worker_role.md",
                                             "worker/initial_work.md"))
        assert [type(t) for t in call["tools"]] == [WorkspaceCodeExec, FetchTool, SearchTool, ViewImageToolProvider]
        assert call["tools"][0]._shell_timeout == 600
        assert expected == [[deliverable]]
        assert [t.name for t in call["finish_tools"]] == ["finish", "abandon_task_finish"]
        assert (call["max_turns"], call["turns_remaining_warning_threshold"], call["context_summarization_cutoff"]) == \
            (250, 50, 0.7)
        assert result.status == "ok" and result.artifact == Path(tmp) / "a0" / "current_artifact"

        revise = Action("worker_revise", "revise", None, (_contribution("a"), _contribution("b")))
        _, calls, _ = _run(runtime, revise, Path(tmp) / "r1", (*_finish(deliverable), {}))
        assert calls[0]["prompt"].endswith(_tail(2, f"- `{deliverable}`", "worker/worker_role.md",
                                                 "worker/worker_revision.md"))
        # abandon fails; at the turn limit the snapshot counts only if every deliverable exists
        for box, script, status in (("x1", ("AbandonParams", {"reason": "r"}, {}), "model_failure"),
                                    ("x2", (None, None, {deliverable: "x"}), "turn_limit_best_available"),
                                    ("x3", (None, None, {}), "model_failure")):
            assert _run(runtime, Action("w", "initial_work", None), Path(tmp) / box, script)[0].status == status


def test_stakeholder_prompts_tools_and_status():
    feedback = "collaboration_workspace/colleague_feedback.md"
    proposal = "collaboration_workspace/improvement_proposal.md"
    with tempfile.TemporaryDirectory() as tmp:
        runtime, _ = _runtime(Path(tmp), ())
        action = Action("a", "feedback", "direct_manager", (_contribution("z"),))
        text = "# Handoff Summary\n\nTotals miss row 7.\n\n# Critical Issues\n\nRow 7.\n"
        result, calls, expected = _run(runtime, action, Path(tmp) / "a", (*_finish(feedback), {feedback: text}))
        call = calls[0]
        assert call["system_prompt"] == rt.ACTION_SYSTEM_PROMPT
        assert call["prompt"].endswith(_tail(1, f"- `{feedback}`", "roles/direct_manager.md", "actions/feedback.md",
                                             "This task has no reference files."))
        assert [type(t) for t in call["tools"]] == [WorkspaceCodeExec, ViewImageToolProvider]
        assert expected == [[feedback]]
        assert (result.status, result.text) == ("ok", text)

        propose = Action("b", "propose", "technical_customer")
        partial = "# Handoff Summary\n\nAdd a check.\n\n# Perspective and Needs\n\nAudit.\n\n# Identified Gaps\n\n"
        result, calls, _ = _run(runtime, propose, Path(tmp) / "b", (*_finish(proposal), {proposal: partial}))
        assert calls[0]["prompt"].endswith(_tail(0, f"- `{proposal}`", "roles/technical_customer.md",
                                                 "actions/propose.md", "This task has no reference files."))
        assert result.status == "protocol_failure"                 # empty Identified Gaps, no Prioritized Proposals
        full = partial + "None.\n\n# Prioritized Proposals\n\n1. Add a check.\n"
        assert _run(runtime, propose, Path(tmp) / "c", (*_finish(proposal), {proposal: full}))[0].status == "ok"
        # the output file starts empty, so a session that never writes it publishes nothing
        result, _, _ = _run(runtime, propose, Path(tmp) / "d", (None, None, {}))
        assert (result.status, result.text) == ("model_failure", "")


# ---- Loop-In -------------------------------------------------------------------

def test_loop_in_driver_phases():
    summary = "# Handoff Summary\n\nI consulted the internal auditor; we agreed to flag row 7.\n"
    script = [("PartnerSelectionParams", {"partner_display_name": "Chief of staff", "opening_message": "x"}, {}),
              ("PartnerSelectionParams", {"partner_display_name": "Internal auditor",
                                          "opening_message": "Please check the totals."}, {}),
              ("TeamMessageParams", {"message": "Row 7 does not tie out."}, {}),
              ("TeamMessageParams", {"message": "Agreed; I will flag it."}, {}),
              (*_finish(OUTPUT), {OUTPUT: summary})]
    action = Action("c", "loop_in", "direct_manager")
    with tempfile.TemporaryDirectory() as tmp:
        runtime, _ = _runtime(Path(tmp))
        result, calls, expected = _run(runtime, action, Path(tmp) / "c", *script)
        references = [REFERENCE.removeprefix("reference_files/")]
    prompt = rt.action_prompt(action, runtime.packet, references)
    assert prompt.endswith(_tail(0, "- Follow the teaming runner's phase-specific tool instructions. After the "
                                    f"discussion, the initiator calls `finish` with `{OUTPUT}`.",
                                 "roles/direct_manager.md", "loop_in/loop_in.md"))
    initiator = f"{rt.INITIATOR_SYSTEM_PROMPT}\n\n{prompts.load('roles/direct_manager.md').strip()}"
    partner = f"{rt.PARTNER_SYSTEM_PROMPT}\n\n{prompts.load('roles/internal_auditor.md').strip()}"
    names = loop_in.partners("direct_manager")
    select = f"{prompt}\n\n" + loop_in.render("loop_in/select.md", {
        "eligible_partner_display_names": "\n".join(f"- {name}" for name in names)})
    retry = (f"{select}\n\n## Corrective Retry\n\nYour prior phase completion was invalid. Correct it now without "
             "restarting the task. Problem: partner_display_name must exactly match an eligible display name: "
             f"{sorted(names)}. Use exactly the completion tool required by this phase.")
    opening = "Message 1 - Direct manager:\nPlease check the totals."
    reply = f"{opening}\n\nMessage 2 - Internal auditor:\nRow 7 does not tie out."

    def discussion(speaker, other, transcript):
        return loop_in.render("loop_in/discussion.md", {
            "participant_display_name": speaker, "other_participant_display_name": other, "round_index": 1,
            "configured_rounds": 1, "transcript": transcript}) + f"\n\n## Original Task\n\n{REQUEST}"

    final = f"{reply}\n\nMessage 3 - Direct manager:\nAgreed; I will flag it."
    assert [(c["system_prompt"], c["prompt"]) for c in calls] == [
        (initiator, select), (initiator, retry),
        (partner, discussion("Internal auditor", "Direct manager", opening)),
        (initiator, discussion("Direct manager", "Internal auditor", reply)),
        (initiator, f"{prompt}\n\n" + loop_in.render("loop_in/summary.md", {"transcript": final}))]
    assert [[t.name for t in c["finish_tools"]][0] for c in calls] == \
        ["select_partner", "select_partner", "send_message", "send_message", "finish"]
    assert all([type(t) for t in c["tools"]] == [WorkspaceCodeExec, ViewImageToolProvider] for c in calls)
    assert all(c["max_turns"] == 30 for c in calls) and expected == [[OUTPUT]]
    assert result.status == "ok" and result.logs["partner"] == "internal_auditor" and result.usage["input"] == 10
    assert result.text.startswith(summary) and "## Round 1 — Internal auditor\n\n> Row 7 does not tie out." in result.text


def test_loop_in_phase_failures():
    select = ("PartnerSelectionParams", {"partner_display_name": "Internal auditor", "opening_message": "Check."}, {})
    abandon = ("AbandonParams", {"reason": "no"}, {})
    action = Action("c", "loop_in", "direct_manager")
    with tempfile.TemporaryDirectory() as tmp:
        runtime, _ = _runtime(Path(tmp))
        result, calls, _ = _run(runtime, action, Path(tmp) / "c", select, abandon, abandon)
        assert result.status == "protocol_failure" and result.logs["phase"] == "round_1_partner"
        assert "Problem: call send_message, not another completion tool." in calls[2]["prompt"]
        assert result.text == ""
        # publication must replace the empty outcome file with exactly one top-level section
        messages = [("TeamMessageParams", {"message": "ok"}, {})] * 2
        two = (*_finish(OUTPUT), {OUTPUT: "# Handoff Summary\n\nx\n\n# Notes\n\ny\n"})
        result, calls, _ = _run(runtime, action, Path(tmp) / "d", select, *messages, (*_finish(OUTPUT), {}), two)
        assert "Problem: empty loop_in summary: " in calls[4]["prompt"]
        assert result.status == "protocol_failure" and result.logs["phase"] == "publish"
        assert "exactly one top-level section: # Handoff Summary" in result.logs["error"]


# ---- submission ----------------------------------------------------------------

def test_submission_copies_every_artifact_file():
    with tempfile.TemporaryDirectory() as tmp:
        artifact, out = Path(tmp) / "artifact", Path(tmp) / "final" / "submission"
        for path in ("Report.docx", "data/Model.xlsx", ".hidden", ".cache/x.bin"):
            (artifact / path).parent.mkdir(parents=True, exist_ok=True)
            (artifact / path).write_bytes(path.encode())
        assert rt.submission(None, "t1", artifact, out) == ["Report.docx", "data/Model.xlsx"]
        assert sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()) == \
            ["Report.docx", "data/Model.xlsx"]
        assert (out / "data/Model.xlsx").read_bytes() == b"data/Model.xlsx"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
