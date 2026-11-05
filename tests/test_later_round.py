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
"""Later-round precedent learning (App. A.3) with a scripted runtime, a fake model and a fake judge.
No API calls."""

import contextlib
import copy
import json
import re
import shutil
import sys
import tempfile
import threading
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from reorg.continual_learning import later_round, ranking
from reorg import llm, planner  # noqa: E402
from reorg.domain import Domain, TaskPacket  # noqa: E402
from reorg.library import Library  # noqa: E402
from reorg.reflection_skill.reflect import ActionResult  # noqa: E402
from reorg.reflection_skill.graph import INPUT, OUTPUT, ROLES, Node, ReflectionSkill, empty_skill  # noqa: E402

CONFIG = json.loads((ROOT / "configs" / "learning.json").read_text(encoding="utf-8"))
PROFILE = llm.ModelProfile(name="fake", provider="gemini", model="fake-model")
STOP = later_round.STOP
LESSON = {"helpful_communication_flows_if_steering": "In this run, a manager review ranked above leaving it.",
          "helpful_communication_flow_features_if_steering": "A sequential handoff coincided with better versions."}


def _skill(skill_id, *steps, parallel=False):
    nodes = tuple(Node(f"n{i}", role, action) for i, (role, action) in enumerate(steps))
    ids = [node.node_id for node in nodes]
    edges = ([(INPUT, i) for i in ids] + [(i, OUTPUT) for i in ids]) if parallel else list(zip([INPUT, *ids], [*ids, OUTPUT]))
    return ReflectionSkill(skill_id, nodes, tuple(edges)).validate()


def _chain(skill_id, roles):
    return _skill(skill_id, *((role, "feedback") for role in roles))


# ---- identity, sampling and groups -------------------------------------------

def test_alias_and_seed_values():
    # Pinned values: the alias names skills and ranking groups, the seed fixes the sampled skills.
    assert later_round.alias("main/data_entry_keyers__task2") == "main_data_entry_keyers__task2__c0e46024102e6563"
    assert later_round.alias("11dcc268-cb07-4d3a-a184-c6d7a19349bc") == \
        "11dcc268-cb07-4d3a-a184-c6d7a19349bc__6837053333b35520"
    assert later_round.task_seed("main/data_entry_keyers__task2", 31415) == 2622499601312901304


def test_sampling_spreads_slots_across_levels():
    roles = list(ROLES)
    skills = ([empty_skill("e0"), empty_skill("e1")]
              + [_chain(f"one_{k}", roles[k:k + 1]) for k in range(4)]
              + [_chain(f"two_{k}", roles[k:k + 2]) for k in range(6)]
              + [_chain("three_0", roles[:3])])
    sampled = later_round.sample(skills, "two_0", 8, seed=7)
    ids = [skill.skill_id for skill in sampled]
    assert len(ids) == 8 == len(set(ids)) and "two_0" not in ids and all(skill.nodes for skill in sampled)
    levels = [len(skill.nodes) for skill in sampled]
    assert sorted(levels[:3]) == [1, 2, 3]                       # the first pass visits every populated level
    order = [n for n in levels[:3] if n != 3]                    # later passes skip the spent level
    assert levels[3:] == (order * 3)[:5]
    assert later_round.sample(skills, "two_0", 8, seed=7) == sampled
    assert len(later_round.sample(skills, "two_0", 50, seed=7)) == 10      # all eligible skills
    assert len({tuple(s.skill_id for s in later_round.sample(skills, "two_0", 8, seed)) for seed in range(6)}) > 1


def test_stop_anchored_groups():
    # Pinned groups for seed 1729 with 5 and 8 candidates.
    assert later_round.stop_groups(["STOP", "s1", "s2", "s3", "s4"], 1729) == [
        ("STOP", "s3", "s2", "s1"), ("STOP", "s4", "s2", "s1"), ("STOP", "s4", "s3", "s1"), ("STOP", "s4", "s3", "s2")]
    assert later_round.stop_groups(["STOP", *(f"s{k}" for k in range(1, 8))], 1729) == [
        ("STOP", "s2", "s3", "s7"), ("STOP", "s5", "s4", "s1"), ("STOP", "s6", "s2", "s3"),
        ("s7", "s5", "s4", "s1"), ("s5", "s4", "s1", "s6"), ("s6", "s2", "s3", "s7")]
    for actions in range(4, 12):
        candidates = [STOP, *(f"skill_{k:02d}" for k in range(actions))]
        groups = later_round.stop_groups(candidates, 1729)
        seen = Counter(c for group in groups for c in group)
        assert all(len(set(group)) == 4 for group in groups) and len({tuple(sorted(g)) for g in groups}) == len(groups)
        assert set(seen) == set(candidates) and min(seen.values()) >= 3
        assert all(any(STOP in group and c in group for group in groups) for c in candidates[1:])
        assert ranking.robust(candidates, groups)
    group = ranking.present("gate3_t__0123456789abcdef", later_round.stop_groups(["STOP", "s1", "s2", "s3", "s4"], 1729),
                            f"{later_round.SCHEDULER}:1729")[0]
    assert group == ranking.Group("gate3_t__0123456789abcdef_group_001", ("STOP", "s3", "s2", "s1"),
                                  {"STOP": "artifact_A", "s2": "artifact_C", "s3": "artifact_D", "s1": "artifact_B"},
                                  ("artifact_A", "artifact_C", "artifact_D", "artifact_B"))


def test_collapse_identical_artifacts():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for name, text in {"STOP": "v0", "a": "v1", "b": "v1", "c": "v0", "d": "v2"}.items():
            (tmp / name / "sub").mkdir(parents=True)
            (tmp / name / "sub" / "memo.md").write_text(text)
        (tmp / "d" / "extra.txt").write_text("v1")              # a file set differs, not only contents
        artifacts = {name: tmp / name for name in ("STOP", "a", "b", "c", "d")}
        assert later_round.collapse(artifacts) == {"STOP": "STOP", "c": "STOP", "a": "a", "b": "a", "d": "d"}


def test_deblinding_and_lesson_validation():
    labels = {"x": "artifact_A", "y": "artifact_B", "z": "artifact_C", STOP: "artifact_D"}
    titles = {"x": "X1", "y": "Y1", "z": "Z1", STOP: "Leaving the work unchanged"}
    # A bare capital letter is replaced too, an article included.
    assert later_round.deblind(" Artifacts A and B beat artifact_C; D lags. A clear call.\n", labels, titles) == \
        "X1, Y1 beat Z1; Leaving the work unchanged lags. X1 clear call."
    later_round.check_lesson(LESSON)
    for leak in ("try flow_004", "artifact_C won", "group_001 tied", "/tmp/x", "a" * 64):
        with contextlib.suppress(ValueError):
            later_round.check_lesson({**LESSON, "helpful_communication_flows_if_steering": leak})
            raise AssertionError(leak)


def test_precedent_document():
    state = {"task_summary": "Build the report.", "what_was_suggested": ["Manager: cap bins.", "Auditor: trace."],
             "what_was_done": []}
    entries = [{"task_id": "z-task", "state": state, "continue_fitness": 0.625, "lesson": LESSON},
               {"task_id": "main/a task", "state": {**state, "what_was_done": ["Owner drafted it."]},
                "continue_fitness": 1.0, "lesson": LESSON}]
    text = later_round.precedents(entries)
    assert text.startswith("# Guidance and examples from earlier work states\n\n"
                           "## How to choose another communication flow\n\n- Base the recommendation")
    assert text.endswith("no causal or held-out generalization claim.\n") and "\n\n\n" not in text
    examples = text.split("## Examples from observed work states\n", 1)[1]
    assert examples == (
        "\n### state 1\n\n"                                      # ordered by alias: main_a_task__... < z-task__...
        "- Current work state: Build the report. Suggested: Manager: cap bins. Auditor: trace. Done: Owner drafted it.\n"
        "- Leaving the work unchanged: continuation fitness `1.0` in this frozen pool.\n"
        f"- Helpful communication flows if steering: {LESSON['helpful_communication_flows_if_steering']}\n"
        f"- Helpful communication-flow features if steering: {LESSON['helpful_communication_flow_features_if_steering']}\n"
        f"- Evidence limits: {later_round.PRECEDENT_LIMITS}\n"
        "\n### state 2\n\n"
        "- Current work state: Build the report. Suggested: Manager: cap bins. Auditor: trace. Done: \n"
        "- Leaving the work unchanged: continuation fitness `0.625` in this frozen pool.\n"
        f"- Helpful communication flows if steering: {LESSON['helpful_communication_flows_if_steering']}\n"
        f"- Helpful communication-flow features if steering: {LESSON['helpful_communication_flow_features_if_steering']}\n"
        f"- Evidence limits: {later_round.PRECEDENT_LIMITS}\n")
    assert later_round.precedents([]).endswith("## Examples from observed work states\n")


# ---- one task end to end -----------------------------------------------------

class ScriptedRuntime:
    """Stakeholders hand off advice, except that internal auditors fail. The worker's memo gains
    one sorted line per contribution it receives; advice from a legal reviewer leaves the memo
    unchanged, and a procurement evaluator's makes the worker cut it back to the draft."""

    def __init__(self):
        self.lock, self.calls = threading.Lock(), Counter()

    def stage(self, sandbox, artifact):
        (sandbox / "current_artifact").mkdir(parents=True, exist_ok=True)
        if artifact is not None:
            shutil.copytree(artifact, sandbox / "current_artifact", dirs_exist_ok=True)

    def run(self, action, sandbox, logs):
        with self.lock:
            self.calls[action.action] += 1
        if action.role:
            ok = action.role != "internal_auditor"
            return ActionResult("ok" if ok else "model_failure", text=f"# Handoff Summary\n\n{action.display_name}")
        memo = sandbox / "current_artifact" / "memo.md"
        roles = {c.role for c in action.incoming}
        if action.action == "initial_work" or "procurement_evaluator" in roles:
            memo.write_text("draft\n")
        elif "legal_risk_reviewer" not in roles:
            memo.write_text(memo.read_text() + "".join(sorted(f"{c.role} {c.action}\n" for c in action.incoming)))
        return ActionResult("ok", artifact=sandbox / "current_artifact")


class FakeDomain(Domain):
    name = "fake"

    def __init__(self, data):
        self.data = data

    def load_task(self, task_id):
        return TaskPacket(task_id, f"Write memo {task_id}.", (self.data / "brief.pdf",), ("out/memo.md",))


SKILLS = [
    _chain("b1", ["direct_manager"]),                                             # the retrieved top-1
    _chain("b2", ["internal_auditor"]),                                           # fails
    _chain("b3", ["legal_risk_reviewer"]),                                        # = STOP
    _skill("c1", ("direct_manager", "feedback"), ("experienced_mentor", "propose")),
    _skill("c2", ("direct_manager", "feedback"), ("experienced_mentor", "propose"), parallel=True),   # = c1
    _skill("c3", ("procurement_evaluator", "propose"), ("technical_customer", "feedback")),        # worse
    _chain("d1", ["same_job_teammate", "upstream_requester", "direct_manager"]),
    _chain("d2", ["cross_functional_partner", "implementation_partner", "direct_manager"]),
    _chain("e1", ["same_job_teammate", "upstream_requester", "direct_manager", "experienced_mentor"]),
    empty_skill("a0"),
]


def fake_judge(group, group_dir, judge):
    """Ranks by memo length; equal lengths tie."""
    files = group_dir / "judge_visible" / "artifacts"
    size = {label: len((files / label / "files" / "memo.md").read_text().splitlines()) for label in group.presentation}
    levels = sorted(set(size.values()), reverse=True)
    return {"group_id": group.group_id, "ranking_tiers": [sorted(l for l in size if size[l] == v) for v in levels],
            "hard_invalid": [], "decisive_difference": "Artifact A lists more advice than artifact_B."}


@contextlib.contextmanager
def _fakes():
    calls, judged = [], []

    def complete_json(profile, *, system, prompt, schema, temperature=None, log_path=None, attempts=3, validate=None):
        if system == planner.SYSTEM["first_round"]:
            assert schema["properties"]["recommendations"]["maxItems"] == 1
            alias = re.search(r"### `(graph_\d{3})`\n\n- Communication flow: the worker creates the initial artifact; "
                              r"before revision 1, the Direct manager reviews the current artifact\n", prompt).group(1)
            reply = {"task_analysis": ["memo"], "recommendations": [{"structure_id": alias, "reason": "fits"}]}
        elif system == planner.SYSTEM["projection"]:
            reply = {"what_was_done": ["Task Owner drafted the memo."], "task_summary": "A memo.",
                     "what_was_suggested": ["Direct manager: add totals."]}       # keys out of order
        else:
            assert system == later_round.SYSTEM and schema["title"].startswith("Useful communication")
            reply = LESSON
        if validate:
            validate(reply)
        calls.append(SimpleNamespace(system=system, prompt=prompt, temperature=temperature))
        return reply

    def judge(group, group_dir, settings):
        judged.append(group.group_id)
        return fake_judge(group, group_dir, settings)

    originals = llm.complete_json, ranking.judge_group
    llm.complete_json, ranking.judge_group = complete_json, judge
    try:
        yield calls, judged
    finally:
        llm.complete_json, ranking.judge_group = originals


def test_later_round_learning_end_to_end_and_resume():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "data").mkdir()
        (tmp / "data/brief.pdf").write_bytes(b"%PDF brief")
        library = Library.create(tmp / "frozen", SKILLS)
        library.add_lesson("## `task_001` — Memo\n\n- Domain: Logistics")

        def files(root):
            return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}

        frozen = files(tmp / "frozen")
        runtime = ScriptedRuntime()

        def submission(domain, task_id, artifact_dir, out_dir):
            shutil.copytree(artifact_dir, out_dir)
            return sorted(p.name for p in out_dir.iterdir())

        hooks = SimpleNamespace(make_runtime=lambda domain, task_id, profile, config: runtime, submission=submission)
        domain, out = FakeDomain(tmp / "data"), tmp / "later"
        with _fakes() as (calls, judged):
            records = later_round.run(domain, ["t1"], library=library, out_dir=out, hooks=hooks, profile=PROFILE,
                                      config=CONFIG)
        (record,) = records
        assert record["status"] == "precedent" and record["retrieved"] == ["b1", "fits"]
        assert [c.system for c in calls] == [planner.SYSTEM["first_round"], planner.SYSTEM["projection"],
                                             later_round.SYSTEM] and {c.temperature for c in calls} == {0.2}
        assert "Write memo t1." in calls[0].prompt and "- reference_files" not in calls[0].prompt   # the raw request

        # Eight unused nonempty skills: the top-1 and the empty skill are never sampled.
        statuses = record["statuses"]
        assert sorted(statuses) == ["b2", "b3", "c1", "c2", "c3", "d1", "d2", "e1"]
        assert {n for n, s in statuses.items() if s != "succeeded"} == {"b2"} and statuses["b2"] == "model_quality_failed"
        assert record["representative"] == {"STOP": "STOP", "b3": "STOP", "c1": "c1", "c2": "c1", "c3": "c3",
                                            "d1": "d1", "d2": "d2", "e1": "e1"}
        ranked = [name for tier in record["tiers"] for name in tier]   # d1 and d2 tie in every group, not in the fit
        assert ranked[0] == "e1" and set(ranked[1:3]) == {"d1", "d2"} and ranked[3:] == ["c1", "STOP", "c3"]
        assert record["precedent"]["continue_fitness"] == 0.8
        assert runtime.calls["initial_work"] == 1 and runtime.calls["revise"] == 9    # the parent's and 8 branches'

        task_dir = out / "tasks" / "t1"
        assert all(g.startswith(f"gate3_{later_round.alias('t1')}_group_") for g in judged)
        visible = next((task_dir / "ranking").iterdir()) / "judge_visible"
        assert (visible / "task.md").read_text() == "Write memo t1.\n"
        assert json.loads((visible / "deliverable_contract.json").read_text()) == {
            "expected_count": 1, "expected_basenames": ["memo.md"], "expected_suffixes": [".md"],
            "visibility_policy": "artifact_only_networkless"}
        artifacts = {p.name for p in (task_dir / "ranking").glob("*/judge_visible/artifacts/*")}
        assert len(list((task_dir / "ranking").iterdir())) >= 3 and artifacts <= {f"artifact_{x}" for x in "ABCD"}

        prompt = calls[-1].prompt
        assert prompt == (task_dir / "lesson/prompt.md").read_text()
        assert '"task_summary": "A memo.",\n  "what_was_suggested"' in prompt
        sampled = [s for s in later_round.sample(SKILLS, "b1", 8, later_round.task_seed("t1", 31415))]
        title = {s.skill_id: f"Communication approach {k}" for k, s in enumerate(sampled, 1)}
        assert f"Leaving the work unchanged = {title['b3']}" in prompt
        assert " = ".join(title[n] for n in ("c1", "c2")) in prompt
        assert ("Among 5 artifact-distinct steering results, 4 ranked above leaving the work unchanged, 0 tied it, "
                "and 1 ranked below it. The programmatic continuation measure is 0.8.") in prompt
        assert f"Excluded from quality ranking: {title['b2']} (model_quality_failed)." in prompt
        assert f"### {title['c1']}\n- On the current work, the Direct manager reviews the current artifact, then " \
               "the Experienced mentor proposes improvements. After this communication" in prompt
        assert "artifact_" not in prompt.split("## Comparison results", 1)[1].split("## Instructions", 1)[0]

        # (L, M) is copied unchanged except for the precedents.
        copied = files(out / "library")
        assert copied.pop("precedents.md") and copied == frozen == files(tmp / "frozen")
        precedents = Library(out / "library").precedents()
        assert precedents == later_round.precedents([record["precedent"]]) and "### state 1" in precedents
        assert "continuation fitness `0.8`" in precedents and "Done: Task Owner drafted the memo." in precedents

        # A finished task reruns without calls; an unfinished one reuses every finished step.
        executed = sum(runtime.calls.values())
        (task_dir / "record.json").unlink()
        with _fakes() as (calls, judged):
            again = later_round.run(domain, ["t1"], library=library, out_dir=out, hooks=hooks, profile=PROFILE,
                                    config=CONFIG)
        assert again == records and not calls and not judged and sum(runtime.calls.values()) == executed
        assert Library(out / "library").precedents() == precedents

        # Too few distinct artifacts: no ranking and no precedent.
        strict = copy.deepcopy(CONFIG)
        strict["later_round"]["min_distinct_artifacts"] = 7
        with _fakes() as (calls, judged):
            (skipped,) = later_round.run(domain, ["t1"], library=library, out_dir=tmp / "strict", hooks=hooks,
                                         profile=PROFILE, config=strict)
        assert skipped["status"] == "too_few_artifacts" and not judged and len(calls) == 2
        assert Library(tmp / "strict/library").precedents() == later_round.precedents([])


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
