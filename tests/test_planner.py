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
"""Library, planner and the Eq. 1 loop with a fake model and a scripted runtime. No API calls."""

import contextlib
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from reorg.reflection_skill import chronicle as chron  # noqa: E402
from reorg import llm, planner  # noqa: E402
from reorg.domain import Domain, TaskPacket  # noqa: E402
from reorg.inference import run_task  # noqa: E402
from reorg.library import LESSONS_HEADER, Library  # noqa: E402
from reorg.reflection_skill.reflect import ActionResult  # noqa: E402
from reorg.reflection_skill.graph import INPUT, OUTPUT, Node, ReflectionSkill, empty_skill  # noqa: E402

CONFIG = json.loads((ROOT / "configs" / "inference.json").read_text(encoding="utf-8"))
PROFILE = llm.ModelProfile(name="fake", provider="gemini", model="fake-model")
PRECEDENTS = "# Guidance and examples from earlier work states\n\n## How to choose another communication flow\n\n- Example.\n"
STATE = {"task_summary": "A memo is drafted.", "what_was_suggested": [], "what_was_done": ["Task Owner drafted it."]}


def _skill(skill_id, *steps, parallel=False):
    """Stakeholder steps (role, action) in sequence, or all in parallel."""
    nodes = tuple(Node(f"n{i}", role, action) for i, (role, action) in enumerate(steps))
    ids = [node.node_id for node in nodes]
    if parallel:
        edges = [(INPUT, i) for i in ids] + [(i, OUTPUT) for i in ids]
    else:
        edges = list(zip([INPUT, *ids], [*ids, OUTPUT]))
    return ReflectionSkill(skill_id, nodes, tuple(edges)).validate()


def _library(root):
    seeds = [empty_skill("seed_1_empty"), _skill("seed_2_feedback", ("direct_manager", "feedback")),
             _skill("seed_3_propose", ("direct_manager", "propose")),
             _skill("seed_4_loop_in", ("direct_manager", "loop_in")),
             _skill("seed_5_feedback_propose", ("direct_manager", "feedback"), ("direct_manager", "propose"))]
    library = Library.create(root / "library", seeds)
    for role in ("same_job_teammate", "upstream_requester", "cross_functional_partner", "technical_customer",
                 "implementation_partner", "procurement_evaluator", "legal_risk_reviewer", "internal_auditor"):
        library.add(_skill(f"t1_{role}", (role, "feedback")), "task_1", {})
    library.write_precedents(PRECEDENTS)
    return library


@contextlib.contextmanager
def _model(**replies):
    """Patch llm.complete_json: each planner call is answered by replies[kind](prompt) and checked by `validate`."""
    calls = []

    def fake(profile, *, system, prompt, schema, temperature=None, log_path=None, attempts=3, validate=None):
        kind = next(key for key, text in planner.SYSTEM.items() if text == system)
        reply = replies[kind](prompt)
        if validate:
            validate(reply)
        calls.append(SimpleNamespace(kind=kind, prompt=prompt, temperature=temperature))
        return reply

    original, llm.complete_json = llm.complete_json, fake
    try:
        yield calls
    finally:
        llm.complete_json = original


def _rank_first(library, skill_id):
    ids = [skill.skill_id for skill in library.skills()]
    others = [skill for skill in ids if skill != skill_id]
    ranked = [skill_id, *others][:CONFIG["first_round_top_k"]]
    return lambda prompt: {"task_analysis": ["needs a manager check"],
                           "recommendations": [{"structure_id": f"graph_{ids.index(s) + 1:03d}", "reason": "fits"}
                                               for s in ranked]}


def _stop_first(prompt):
    flows = re.findall(r"### `(flow_\d{3})`", prompt)[:CONFIG["later_round_flows"]]
    return {"analysis": "Mostly done.",
            "recommendations": [{"option_id": planner.STOP, "reason": "looks complete"}]
            + [{"option_id": flow, "reason": "focused"} for flow in flows]}


def _score(value):
    return lambda prompt: {"analysis": " evidence ", "task_completion_score": value}


def _rejects(check, *args):
    try:
        check(*args)
    except ValueError:
        return True
    return False


# ---- library -----------------------------------------------------------------

def test_library_round_trip():
    with tempfile.TemporaryDirectory() as tmp:
        library = _library(Path(tmp))
        renamed = ReflectionSkill("t2_copy", (Node("x", "legal_risk_reviewer", "feedback"),),
                                  ((INPUT, "x"), ("x", OUTPUT)))
        assert library.add(renamed, "task_2", {"candidate": "c1"}) == ("reinforced", "t1_legal_risk_reviewer")
        assert library.add(_skill("t2_new", ("internal_auditor", "propose")), "task_2", {}) == ("promoted", "t2_new")
        try:
            library.add(_skill("t2_new", ("internal_auditor", "loop_in")), "task_3", {})
            raise AssertionError("a skill ID names one skill")
        except FileExistsError:
            pass

        reloaded = Library(library.root)
        assert [s.to_dict() for s in reloaded.skills()] == [s.to_dict() for s in library.skills()]
        assert [s.skill_id for s in reloaded.skills()] == sorted(s.skill_id for s in reloaded.skills())
        assert reloaded.skill("seed_2_feedback").fingerprint() == _skill("z", ("direct_manager", "feedback")).fingerprint()
        assert reloaded.find(renamed.fingerprint()) == "t1_legal_risk_reviewer" and reloaded.find("0" * 64) is None
        observed = reloaded.observations()
        assert observed["t1_legal_risk_reviewer"] == 2 and observed["seed_1_empty"] == 0 and observed["t2_new"] == 1
        last = reloaded.records()[-1]
        assert last["event"] == "promoted" and last["task_id"] == "task_2" and set(last) == {
            "skill_id", "fingerprint", "event", "task_id", "evidence"}

        assert reloaded.lessons() == LESSONS_HEADER + "\n\nNo completed-task observations are available yet.\n"
        reloaded.add_lesson("## `task_001` — Memo\n\n- Domain: Legal\n")
        reloaded.add_lesson("## `task_002` — Model")
        assert reloaded.lessons() == LESSONS_HEADER + "\n\n## `task_001` — Memo\n\n- Domain: Legal\n\n## `task_002` — Model\n"
        assert reloaded.precedents() == PRECEDENTS
        copy = reloaded.snapshot(Path(tmp) / "snapshot")
        assert copy.records() == reloaded.records() and copy.lessons() == reloaded.lessons()


# ---- prompts -----------------------------------------------------------------

def test_communication_flow_describes_stages():
    skill = ReflectionSkill("s", (Node("a", "internal_auditor", "feedback"), Node("b", "direct_manager", "feedback"),
                                  Node("c", "legal_risk_reviewer", "propose")),
                            ((INPUT, "a"), (INPUT, "b"), ("a", "c"), ("b", "c"), ("c", OUTPUT))).validate()
    assert planner.communication_flow(skill) == (
        "the Direct manager reviews the current artifact in parallel with the Internal auditor reviews the current "
        "artifact, then the Legal and risk reviewer proposes improvements")
    assert planner.communication_flow(empty_skill()) == ""


def test_task_query():
    root = Path("data") / "gdpval"
    packet = TaskPacket("t", "  Draft the memo.\n", (root / "reference_files/ab/brief.pdf",))
    assert planner.task_query(packet, root) == (
        "Task:\nDraft the memo.\n\nReference files:\n- reference_files/ab/brief.pdf\n\n"
        "Expected deliverables:\n- None declared\n")


def test_prompt_rendering_is_deterministic():
    with tempfile.TemporaryDirectory() as tmp:
        library = _library(Path(tmp))
        skills, observed = library.skills(), library.observations()
        first, aliases = planner.first_round_prompt("Task:\nDraft.\n", skills, observed, library.lessons(), 5)
        again, _ = planner.first_round_prompt("Task:\nDraft.\n", list(Library(library.root).skills()),
                                              Library(library.root).observations(), library.lessons(), 5)
        assert first == again and first.endswith("exactly 5 unique available graph aliases.\n")
        assert list(aliases.values()) == [s.skill_id for s in skills] and "<<" not in first
        assert ("### `graph_001`\n\n- Communication flow: the worker creates the initial artifact; revision 1 "
                "proceeds without colleague communication\n- Prior top-three promotion or reinforcement "
                "observations: 0") in first
        assert "- **Legal and risk reviewer**: " in first and "legal_risk_reviewer" not in first
        assert planner.HELPFUL_TASKS in first and LESSONS_HEADER in first

        flows = planner.later_round_flows(skills, [library.skill("seed_2_feedback")])
        later = planner.later_round_prompt(STATE, flows, library.precedents())
        assert later == planner.later_round_prompt(STATE, flows, library.precedents()) and "<<" not in later
        assert "## Suggestions so far\n\n- None recorded.\n\n## Work done so far\n\n- Task Owner drafted it.\n" in later
        assert later.count("- After this communication, the Task Owner revises the current work once.") == len(flows)
        assert later.endswith("Return JSON only, matching the supplied schema.\n") and PRECEDENTS in later

        completion = planner.completion_prompt(STATE)
        assert completion == planner.completion_prompt(STATE)
        assert "## Work done so far\n\n- Task Owner drafted it.\n\nReturn JSON" in completion
        projection = planner.projection_prompt("Draft.", chron.INITIAL)
        assert projection == planner.projection_prompt("Draft.", chron.INITIAL) and chron.INITIAL in projection


# ---- selection rules ---------------------------------------------------------

def test_later_round_flows_exclude_used_ids_and_fingerprints():
    with tempfile.TemporaryDirectory() as tmp:
        library = _library(Path(tmp))
        nonempty = [s.skill_id for s in library.skills() if s.nodes]
        same_structure = _skill("elsewhere", ("direct_manager", "propose"))        # fingerprint of seed_3_propose
        flows = planner.later_round_flows(library.skills(), [library.skill("seed_2_feedback"), same_structure])
        listed = {skill.skill_id: alias for alias, skill in flows}
        assert "seed_1_empty" not in listed and "seed_2_feedback" not in listed and "seed_3_propose" not in listed
        # aliases are numbered over all nonempty skills, so excluded ones leave gaps
        assert all(alias == f"flow_{nonempty.index(skill_id) + 1:03d}" for skill_id, alias in listed.items())
        assert len(flows) == len(nonempty) - 2


def test_stop_ranked_first_still_runs_the_top_skill():
    with tempfile.TemporaryDirectory() as tmp:
        library = _library(Path(tmp))
        used = [library.skill("seed_2_feedback")]
        with _model(later_round=_stop_first) as calls:
            skill_id, ranking = planner.select_later_round(PROFILE, STATE, library, used, count=9, temperature=0.2)
        first_flow = planner.later_round_flows(library.skills(), used)[0][1].skill_id
        assert ranking["recommendations"][0]["option_id"] == planner.STOP and skill_id == first_flow
        assert calls[0].temperature == 0.2


def test_reply_validation():
    allowed = [f"flow_{i:03d}" for i in range(1, 11)]
    good = [{"option_id": planner.STOP, "reason": "r"}] + [{"option_id": a, "reason": "r"} for a in allowed[:9]]
    planner.check_later_round({"analysis": "a", "recommendations": good}, allowed, 9)
    no_stop = good[1:] + [{"option_id": allowed[9], "reason": "r"}]
    two_stops = good[:9] + [{"option_id": planner.STOP, "reason": "r"}]
    unknown = good[:9] + [{"option_id": "flow_099", "reason": "r"}]
    duplicate = good[:9] + [good[1]]
    for rows in (no_stop, two_stops, unknown, duplicate, good[:9]):
        assert _rejects(planner.check_later_round, {"analysis": "a", "recommendations": rows}, allowed, 9)

    graphs = ["graph_001", "graph_002", "graph_003"]
    reply = {"task_analysis": ["x"], "recommendations": [{"structure_id": g, "reason": "r"} for g in graphs[:2]]}
    planner.check_first_round(reply, graphs, 2)
    assert _rejects(planner.check_first_round, {**reply, "recommendations": reply["recommendations"][:1] * 2}, graphs, 2)
    assert _rejects(planner.check_first_round, {**reply, "task_analysis": []}, graphs, 2)
    assert _rejects(planner.check_first_round, {**reply, "extra": 1}, graphs, 2)


# ---- the Eq. 1 loop ----------------------------------------------------------

class ScriptedRuntime:
    """Stakeholders hand off advice; the worker appends its action to memo.md."""

    def __init__(self, fail=()):
        self.fail, self.seen = set(fail), []

    def stage(self, sandbox, artifact):
        (sandbox / "current_artifact").mkdir(parents=True, exist_ok=True)
        if artifact is not None:
            shutil.copytree(artifact, sandbox / "current_artifact", dirs_exist_ok=True)

    def run(self, action, sandbox, logs):
        self.seen.append(action.vertex_id)
        if action.vertex_id in self.fail:
            return ActionResult("model_failure")
        if action.role:
            return ActionResult("ok", text=f"# Handoff Summary\n\n{action.display_name} advice")
        memo = sandbox / "current_artifact" / "memo.md"
        memo.write_text((memo.read_text() if memo.exists() else "") + f"{action.action}\n")
        return ActionResult("ok", artifact=sandbox / "current_artifact")


class FakeDomain(Domain):
    name = "fake"

    def __init__(self, data):
        self.data = data

    def load_task(self, task_id):
        return TaskPacket(task_id, "Write the memo.", (self.data / "reference_files" / "brief.pdf",), ("memo.md",))


def _run(tmp, runtime, scores, **config):
    library = _library(tmp)

    def submission(domain, task_id, artifact_dir, out_dir):
        shutil.copytree(artifact_dir, out_dir)
        return sorted(path.name for path in out_dir.iterdir())

    hooks = SimpleNamespace(make_runtime=lambda domain, task_id, profile, cfg: runtime, submission=submission)
    pending = list(scores)
    with _model(first_round=_rank_first(library, "seed_2_feedback"),
                projection=lambda prompt: STATE,
                completion=lambda prompt: _score(pending.pop(0))(prompt),
                later_round=_stop_first) as calls:
        result = run_task(FakeDomain(tmp / "data"), "task_1", hooks=hooks, library=library, worker=PROFILE,
                          planner_profile=PROFILE, config={**CONFIG, **config}, task_dir=tmp / "run" / "task_1")
    return result, calls, library, tmp / "run" / "task_1"


def test_completion_gate_stops_the_loop():
    with tempfile.TemporaryDirectory() as tmp:
        result, calls, _, task_dir = _run(Path(tmp), ScriptedRuntime(), scores=[9])
        assert result["stop"] == "completion_gate" and result["status"] == "complete"
        assert [row.get("skill_id") for row in result["rounds"]] == ["seed_2_feedback", None]
        assert [call.kind for call in calls] == ["first_round", "projection", "completion"]
        assert "- reference_files/brief.pdf" in calls[0].prompt and "- memo.md" in calls[0].prompt
        assert (task_dir / "final/submission/memo.md").read_text() == "initial_work\nrevise\n"
        assert (task_dir / "a0/artifact/memo.md").is_file() and (task_dir / "round_2/planner.json").is_file()
        assert not (task_dir / "round_2/round.json").exists()
        assert json.loads((task_dir / "result.json").read_text()) == result


def test_later_rounds_exclude_used_skills():
    with tempfile.TemporaryDirectory() as tmp:
        result, calls, library, task_dir = _run(Path(tmp), ScriptedRuntime(), scores=[5, 8])
        chosen = [row["skill_id"] for row in result["rounds"]]
        assert result["stop"] == "max_rounds" and len(set(chosen)) == 3 and chosen[0] == "seed_2_feedback"
        assert chosen[1] == planner.later_round_flows(library.skills(), [library.skill(chosen[0])])[0][1].skill_id
        assert [row["completion_score"] for row in result["rounds"]] == [None, 5, 8]
        assert all(call.temperature == CONFIG["planner_temperature"]["gemini"] for call in calls)
        assert (task_dir / "final/submission/memo.md").read_text() == "initial_work\nrevise\nrevise\nrevise\n"
        # the round-3 prompt no longer lists either used skill
        listed = re.findall(r"### `(flow_\d{3})`", calls[-1].prompt)
        flows = dict(planner.later_round_flows(library.skills(), []))
        assert not {flows[alias].skill_id for alias in listed} & set(chosen[:2])


def test_non_strict_round_ends_the_loop_and_zero_rounds_is_the_worker_alone():
    with tempfile.TemporaryDirectory() as tmp:
        result, _, _, task_dir = _run(Path(tmp), ScriptedRuntime(fail={"n0"}), scores=[])
        assert result["stop"] == "round_degraded" and len(result["rounds"]) == 1 and not (task_dir / "round_2").exists()
    with tempfile.TemporaryDirectory() as tmp:
        result, calls, _, task_dir = _run(Path(tmp), ScriptedRuntime(), scores=[], max_rounds=0)
        assert result["stop"] == "max_rounds" and not calls and result["submission"] == ["memo.md"]
    with tempfile.TemporaryDirectory() as tmp:
        result, calls, _, _ = _run(Path(tmp), ScriptedRuntime(fail={"worker_initial"}), scores=[])
        assert result["stop"] == "initial_work_failed" and result["status"] == "incomplete" and not calls


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
