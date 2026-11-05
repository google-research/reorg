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
"""Library learning (Alg. 1) with a scripted runtime, a fake model and a fake judge. No API calls."""

import contextlib
import itertools
import json
import math
import random
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

from reorg.continual_learning import aggregate, knowledge, learn, mutate, ranking, search
from reorg import llm, planner  # noqa: E402
from reorg.domain import Domain, TaskPacket  # noqa: E402
from reorg.library import Library  # noqa: E402
from reorg.reflection_skill.reflect import ActionResult  # noqa: E402
from reorg.continual_learning.seeds import SEEDS  # noqa: E402
from reorg.reflection_skill.graph import INPUT, MAX_NODES, OUTPUT, ROLES, Node, ReflectionSkill  # noqa: E402

CONFIG = json.loads((ROOT / "configs" / "learning.json").read_text(encoding="utf-8"))
PROFILE = llm.ModelProfile(name="fake", provider="gemini", model="fake-model")
REPLY = {"task_summary": {"domain": "Logistics", "high_level_task": "Write the memo",
                            "requested_output": "A memo"},
           "structure_helpfulness_analysis": "In this run, reviews coincided with better memos.",
           "critical_feature_analysis": "Sequential review may have helped."}


def _skill(skill_id, *steps, parallel=False):
    nodes = tuple(Node(f"n{i}", role, action) for i, (role, action) in enumerate(steps))
    ids = [node.node_id for node in nodes]
    edges = ([(INPUT, i) for i in ids] + [(i, OUTPUT) for i in ids]) if parallel else list(zip([INPUT, *ids], [*ids, OUTPUT]))
    return ReflectionSkill(skill_id, nodes, tuple(edges)).validate()


# ---- seeds and mutation ------------------------------------------------------

def test_seeds():
    assert [s.skill_id for s in SEEDS] == sorted(s.skill_id for s in SEEDS)
    assert len({s.fingerprint() for s in SEEDS}) == 5 and not SEEDS[0].nodes
    assert {node.role for s in SEEDS for node in s.nodes} == {"direct_manager"}
    assert [n.action for n in SEEDS[4].nodes] == ["feedback", "propose"]
    assert SEEDS[4].edges == ((INPUT, "direct_manager_feedback_1"), ("direct_manager_feedback_1", "direct_manager_propose_1"),
                              ("direct_manager_propose_1", OUTPUT))


def test_edit_semantics():
    chain = SEEDS[4]
    first = mutate.apply(chain, mutate.Edit("insert_propose", None, "propose", "internal_auditor", "sequential", INPUT), "c")
    assert set(first.edges) == {(INPUT, "internal_auditor_propose_1"), ("internal_auditor_propose_1", "direct_manager_feedback_1"),
                                ("direct_manager_feedback_1", "direct_manager_propose_1"), ("direct_manager_propose_1", OUTPUT)}
    side = mutate.apply(chain, mutate.Edit("insert_feedback", None, "feedback", "direct_manager", "parallel",
                                           "direct_manager_feedback_1"), "c")
    assert side.nodes[-1].node_id == "direct_manager_feedback_2"           # named after the free index
    assert {(INPUT, "direct_manager_feedback_2"), ("direct_manager_feedback_2", "direct_manager_propose_1")} <= set(side.edges)
    moved = mutate.apply(chain, mutate.Edit("relocate", "direct_manager_feedback_1", placement="sequential",
                                            anchor="direct_manager_propose_1"), "c")
    assert [n.node_id for n in moved.nodes] == ["direct_manager_propose_1", "direct_manager_feedback_1"]
    assert set(moved.edges) == {(INPUT, "direct_manager_propose_1"), ("direct_manager_propose_1", "direct_manager_feedback_1"),
                                ("direct_manager_feedback_1", OUTPUT)}
    # Relocating one of two parallel nodes reconnects INPUT to OUTPUT; apply drops that edge.
    pair = _skill("p", ("direct_manager", "feedback"), ("internal_auditor", "feedback"), parallel=True)
    serial = mutate.apply(pair, mutate.Edit("relocate", "n0", placement="sequential", anchor="n1"), "c")
    assert set(serial.edges) == {(INPUT, "n1"), ("n1", "n0"), ("n0", OUTPUT)}
    renamed = mutate.apply(chain, mutate.Edit("replace_role", "direct_manager_propose_1", role="legal_risk_reviewer"), "c")
    assert renamed.edges == chain.edges and renamed.nodes[1] == Node("direct_manager_propose_1", "legal_risk_reviewer", "propose")


def test_mutation_legality():
    rng, skill = random.Random(0), SEEDS[0]
    for _ in range(MAX_NODES + 1):
        n, edits = len(skill.nodes), mutate.legal_edits(skill)
        inserts = [e for f in ("insert_feedback", "insert_propose", "insert_loop_in") for e in edits[f]]
        assert len(inserts) == (3 * len(ROLES) * (1 + 2 * n) if n < MAX_NODES else 0)
        assert len(edits["replace_role"]) == n * (len(ROLES) - 1) and len(edits["relocate"]) == n * (2 * n - 1)
        assert all(e.anchor != e.node for e in edits["relocate"])
        for edit in (e for family in mutate.FAMILIES for e in edits[family]):
            child = mutate.apply(skill, edit, "child")          # validate() is the legality check
            assert len(child.nodes) <= MAX_NODES and ((INPUT, OUTPUT) not in child.edges) == bool(child.nodes)
        skill = mutate.apply(skill, rng.choice(inserts or edits["relocate"]), "walk")
    assert len(skill.nodes) == MAX_NODES and set(mutate.ROLE_ORDER) == set(ROLES)


def test_mutation_weights():
    skill = SEEDS[4]
    legal = [e for family in mutate.FAMILIES for e in mutate.legal_edits(skill)[family]]
    drawn = list(mutate.draws(skill, 7))
    assert len(drawn) == len(legal) and set(drawn) == set(legal)            # without replacement
    first = Counter(next(mutate.draws(skill, seed)).family for seed in range(4000))
    total = sum(mutate.WEIGHTS.values())
    assert all(abs(first[f] / 4000 - w / total) < 0.03 for f, w in mutate.WEIGHTS.items())
    full = mutate.apply(SEEDS[1], mutate.Edit("insert_feedback", None, "feedback", "internal_auditor", "sequential", INPUT), "f")
    for _ in range(MAX_NODES - 2):
        full = mutate.apply(full, mutate.Edit("insert_feedback", None, "feedback", "internal_auditor", "sequential", INPUT), "f")
    capped = Counter(next(mutate.draws(full, seed)).family for seed in range(2000))   # no insertion at the cap
    assert set(capped) == {"relocate", "replace_role"} and abs(capped["relocate"] / 2000 - 0.75) < 0.04
    # A first search step from the manager-Feedback seed: its second draw relocates the only node
    # onto itself, a duplicate that still consumes c002.
    seen = {s.fingerprint() for s in SEEDS}
    children = mutate.children(SEEDS[1], "retrieved_004", 1, limit=3, seen=seen)
    assert [c for c, _, _ in children] == ["r001_retrieved_004_c001", "r001_retrieved_004_c003", "r001_retrieved_004_c004"]
    assert [(e.family, e.role, e.placement) for _, _, e in children] == [
        ("insert_feedback", "experienced_mentor", "parallel"), ("replace_role", "legal_risk_reviewer", None),
        ("insert_feedback", "technical_customer", "parallel")]
    assert {child.fingerprint() for _, child, _ in children} <= seen and len(seen) == 8


# ---- ranking and aggregation -------------------------------------------------

def _appearances(groups):
    return Counter(c for g in groups for c in g.candidates)


def test_group_construction():
    five = ranking.schedule("initial", [f"retrieved_{i:03d}" for i in range(1, 6)])
    assert len(five) == 4 and min(_appearances(five).values()) == 3 and max(_appearances(five).values()) == 4
    assert five[0] == ranking.Group("initial_group_001", ("retrieved_005", "retrieved_003", "retrieved_004", "retrieved_001"),
                                    {"retrieved_005": "artifact_A", "retrieved_004": "artifact_D",
                                     "retrieved_003": "artifact_B", "retrieved_001": "artifact_C"},
                                    ("artifact_A", "artifact_D", "artifact_B", "artifact_C"))
    families = {p: [f"{p}_c{k}" for k in range(3)] for p in ("a", "b", "c")}
    twelve = ranking.schedule("search_001", [*families, *itertools.chain(*families.values())], 1729,
                              [(p, c) for p, cs in families.items() for c in cs], families)
    assert len(twelve) == 9 and set(_appearances(twelve).values()) == {3}
    pairs = {frozenset(pair) for g in twelve for pair in itertools.combinations(g.candidates, 2)}
    assert all(frozenset((p, c)) in pairs for p, cs in families.items() for c in cs)
    eight, required = [f"k{i}" for i in range(8)], [("k0", "k5"), ("k1", "k6"), ("k2", "k7")]
    groups = ranking.schedule("search_002", eight, 1729, required, {"k0": ["k5"]})
    counts = _appearances(groups)
    assert min(counts.values()) >= 3 and max(counts.values()) - min(counts.values()) <= 1
    covered = {frozenset(pair) for g in groups for pair in itertools.combinations(g.candidates, 2)}
    assert all(frozenset(pair) in covered for pair in required) and ranking.robust(eight, [g.candidates for g in groups])
    assert groups == ranking.schedule("search_002", eight[::-1], 1729, required, {"k0": ["k5"]})
    for group in groups + five:
        assert sorted(group.presentation) == list(ranking.LABELS) and set(group.labels) == set(group.candidates)
        assert [group.labels[c] for c in group.candidates] != [] and group.group_id in ranking.instruction(group)


def test_aggregation_known_case():
    # An initial tournament: four groups over five candidates, one candidate hard-invalid once.
    groups = [{"candidates": ["r5", "r3", "r4", "r1"], "tiers": [["r5"], ["r4"], ["r3"], ["r1"]], "hard_invalid": []},
              {"candidates": ["r5", "r2", "r4", "r1"], "tiers": [["r5"], ["r4"], ["r1"], ["r2"]], "hard_invalid": []},
              {"candidates": ["r5", "r2", "r3", "r1"], "tiers": [["r1"], ["r3"], ["r5"], ["r2"]], "hard_invalid": []},
              {"candidates": ["r5", "r2", "r3", "r4"], "tiers": [["r4"], ["r3"], ["r5"], ["r2"]], "hard_invalid": ["r2"]}]
    result = aggregate.aggregate(groups)
    stored = {"r1": 0.16300194197917492, "r2": 0.0672938347649597, "r3": 0.20188487082811055,
              "r4": 0.31060390397148663, "r5": 0.25721544845626826}
    assert all(abs(result["strengths"][c] - stored[c]) < 1e-12 for c in stored)
    assert abs(result["tie"] - 0.29772638868933315) < 1e-12 and result["fit"]["iterations"] == 220
    assert result["tiers"] == [["r4"], ["r5"], ["r3"], ["r1"], ["r2"]] and result["hard_invalid"] == ["r2"]
    assert result["statistics"]["r5"] == {"appearances": 4, "distinct_opponents": 4, "wins": 8, "ties": 0, "losses": 4}
    assert abs(sum(result["log_strengths"].values())) < 1e-12

    # Two candidates tied everywhere share a tier; the fit is a stationary point of the objective.
    tied = [{"candidates": list(g), "tiers": [["a", "b"], [c for c in g if c not in "ab"]], "hard_invalid": []}
            for g in (("a", "b", "c", "d"), ("a", "b", "c", "e"), ("a", "b", "d", "e"))]
    tied.append({"candidates": ["c", "d", "e", "a"], "tiers": [["a"], ["c"], ["d"], ["e"]], "hard_invalid": []})
    tied.append({"candidates": ["c", "d", "e", "b"], "tiers": [["b"], ["c"], ["d"], ["e"]], "hard_invalid": []})
    result = aggregate.aggregate(tied)
    assert result["tiers"][0] == ["a", "b"] and result["fit"]["converged"]
    theta, eta = result["log_strengths"], math.log(result["tie"])
    observations = aggregate.pairs(tied)

    def objective(values, tie):
        loss = 0.0
        for left, right, outcome in observations:
            p = aggregate._softmax((values[left], values[right], tie + 0.5 * (values[left] + values[right])))
            loss -= math.log(p[outcome])
        return loss / len(observations) + 0.05 * sum(v * v for v in values.values()) + 0.05 * tie * tie

    step = 1e-6
    for a, b in (("a", "c"), ("c", "e")):                    # directions inside sum(theta) = 0
        plus, minus = dict(theta), dict(theta)
        plus[a], plus[b], minus[a], minus[b] = theta[a] + step, theta[b] - step, theta[a] - step, theta[b] + step
        assert abs(objective(plus, eta) - objective(minus, eta)) / (2 * step) < 1e-5
    assert abs(objective(theta, eta + step) - objective(theta, eta - step)) / (2 * step) < 1e-5


def test_finish_ranking_validation():
    ok = {"group_id": "g", "ranking_tiers": [["artifact_B"], ["artifact_A", "artifact_C", "artifact_D"]],
          "hard_invalid": ["artifact_D"], "decisive_difference": "B cites the log."}
    assert ranking.ranking_error(ranking.RankingFinishParams.model_validate(ok), "g") is None
    for bad in ({"group_id": "h"}, {"ranking_tiers": [["artifact_A", "artifact_B"]]},
                {"ranking_tiers": [["artifact_A", "artifact_A", "artifact_B", "artifact_C", "artifact_D"]]},
                {"hard_invalid": ["artifact_E"]}, {"decisive_difference": " "}):
        assert ranking.ranking_error(ranking.RankingFinishParams.model_validate({**ok, **bad}), "g")
    assert "Hard limit: 100 calls." in ranking.JudgeCodeExec(Path("w"), 600, 100).get_code_exec_tool().description
    assert ranking.INSTALL.search("cd x && pip install pandas") and not ranking.INSTALL.search("python3 parse.py")


# ---- beam, promotion and library decisions -----------------------------------

def _candidates(skills, eligible=lambda c: True):
    out = {}
    for skill in skills:
        candidate = search.Candidate(skill.skill_id, skill)
        candidate.artifact = Path("artifact") if eligible(candidate) else None
        out[skill.skill_id] = candidate
    return out


def test_beam_selection():
    skills = [_skill(f"s{i}", (role, "feedback")) for i, role in enumerate(list(ROLES)[:5])]
    candidates = _candidates(skills)
    by_fp = sorted(candidates, key=lambda c: candidates[c].fingerprint)
    record = {"tiers": [["s0"], ["s1", "s2", "s3"], ["s4"]], "hard_invalid": []}
    picked = [c.candidate_id for c in search.select(record, candidates, 3)]
    assert picked[0] == "s0" and picked[1:] == [c for c in by_fp if c in ("s1", "s2", "s3")][:2]
    assert [c.candidate_id for c in search.select({**record, "hard_invalid": ["s0", "s2"]}, candidates, 3)] == \
        [c for c in by_fp if c in ("s1", "s3")] + ["s4"]
    assert len(search.select(record, candidates, 10)) == 5


def _fake_rank(log):
    """rank(epoch, pool, families): more stakeholder nodes rank higher; equal counts tie."""
    def rank(epoch_id, pool, families=None):
        eligible = [c for c in pool if c.artifact]
        log.append((epoch_id, [c.candidate_id for c in pool], families))
        record = {"epoch_id": epoch_id, "status": "skipped", "candidates": [c.candidate_id for c in pool],
                  "ineligible": [c.candidate_id for c in pool if not c.artifact]}
        if len(eligible) >= search.MIN_POOL:
            levels = sorted({len(c.skill.nodes) for c in eligible}, reverse=True)
            record.update(status="completed", hard_invalid=[], tiers=[
                sorted(c.candidate_id for c in eligible if len(c.skill.nodes) == level) for level in levels])
        return record
    return rank


def test_search_and_promotion():
    def execute(candidates):
        for candidate in candidates:
            candidate.artifact = Path("artifact")

    log = []
    outcome = search.run(list(SEEDS), execute=execute, rank=_fake_rank(log))
    epochs = [row[0] for row in log]
    assert epochs == ["initial", "search_001", "search_002", "search_003"] and outcome.status == "completed"
    assert outcome.initial_order[0] == "retrieved_005" and len(outcome.initial_order) == 5
    for epoch, pool, families in log[1:]:
        beam = [c for c in pool if c in families]
        assert len(beam) == 3 and all(len(children) <= 3 for children in families.values())
        assert set(pool) == set(beam) | {c for children in families.values() for c in children}
    assert len({c.fingerprint for c in outcome.candidates.values()}) == len(outcome.candidates)
    assert all(len(outcome.candidates[f].skill.nodes) >= 4 for f in outcome.finalists)

    record, selected = search.promote(outcome, rank=_fake_rank(log))
    pool = log[-1][1]
    assert log[-1][0] == "promotion" and pool[:3] == outcome.finalists and len(pool) == 5
    assert pool[3:] == [c for c in outcome.initial_order if c not in pool[:3]][:2]
    assert [c.candidate_id for c in selected] == [c.candidate_id for c in search.select(record, outcome.candidates, 3)]

    # Fewer than five eligible retrieved artifacts: no tournament, the retrieval order seeds the beam.
    def partial(candidates):
        for candidate in candidates:
            candidate.artifact = Path("a") if candidate.candidate_id not in ("retrieved_002", "retrieved_004") else None

    log = []
    outcome = search.run(list(SEEDS), execute=partial, rank=_fake_rank(log), iterations=1)
    assert log[0][0] == "initial" and outcome.initial_order == ["retrieved_001", "retrieved_003", "retrieved_005"]
    assert [c for c in log[1][2]] == ["retrieved_001", "retrieved_003", "retrieved_005"]
    outcome = search.run(list(SEEDS), execute=lambda cs: None, rank=_fake_rank([]))
    assert outcome.status == "skipped" and outcome.finalists == []


def test_library_decisions():
    with tempfile.TemporaryDirectory() as tmp:
        library = Library.create(Path(tmp) / "library", SEEDS)
        new = search.Candidate("r001_retrieved_002_c001", _skill("x", ("internal_auditor", "propose")))
        old = search.Candidate("retrieved_002", SEEDS[1])
        rerun = search.Candidate("r002_r001_retrieved_002_c001_c003", _skill("y", ("direct_manager", "loop_in")))
        decision = learn.decide(library, "t1", {"status": "completed"}, [new, old, rerun])
        assert decision["status"] == "updated" and decision["promoted"] == ["t1_r001_retrieved_002_c001"]
        assert decision["reinforced"] == [SEEDS[1].skill_id, SEEDS[3].skill_id]
        assert library.skill("t1_r001_retrieved_002_c001").nodes == new.skill.nodes
        assert library.observations() == Counter({"t1_r001_retrieved_002_c001": 1, SEEDS[1].skill_id: 1, SEEDS[3].skill_id: 1})
        events = len(library.records())
        assert learn.decide(library, "t2", {"status": "completed"}, [new, old])["status"] == "no_update"
        assert learn.decide(library, "t2", {"status": "skipped"}, [])["status"] == "no_update"
        assert len(library.records()) == events


# ---- lessons -----------------------------------------------------------------

def test_lesson_observations_and_prompt():
    a, b = _skill("a", ("direct_manager", "feedback")), _skill("b", ("cross_functional_partner", "propose"))
    child = mutate.apply(a, mutate.Edit("insert_propose", None, "propose", "cross_functional_partner", "sequential",
                                        "n0"), "r001_retrieved_001_c001")
    candidates = {"retrieved_001": search.Candidate("retrieved_001", a), "retrieved_002": search.Candidate("retrieved_002", b),
                  child.skill_id: search.Candidate(child.skill_id, child, "retrieved_001",
                                                   mutate.Edit("insert_propose", None, "propose", "cross_functional_partner",
                                                               "sequential", "n0"))}
    stats = {c: {"appearances": 3, "distinct_opponents": 4} for c in candidates}
    groups = [{"candidates": ["retrieved_001", child.skill_id, "retrieved_002", "z"], "decisive_difference": "B | wins\nclearly",
               "labels": {"retrieved_001": "artifact_C", child.skill_id: "artifact_B", "retrieved_002": "artifact_A", "z": "artifact_D"}}]
    search_record = {"epoch_id": "search_001", "status": "completed", "candidates": list(candidates), "ineligible": [],
                     "tiers": [[child.skill_id], ["retrieved_001", "retrieved_002"]], "hard_invalid": [],
                     "statistics": stats, "groups": groups}
    promotion = {**search_record, "epoch_id": "promotion", "tiers": [["retrieved_002"], [child.skill_id], ["retrieved_001"]]}
    outcome = search.Outcome("completed", candidates, ["retrieved_001", "retrieved_002"], [], [], [search_record])
    rows = knowledge.observations(outcome, promotion)
    assert [(r["stage"], r["relation"]) for r in rows] == [("promotion", "worse"), ("promotion", "worse"), ("search", "better")]
    sections = knowledge.select(rows)
    assert not sections["most_effective_structures"] and len(sections["least_effective_structures"]) == 2
    assert sections["most_effective_edits"][0]["edit"] == "add xfn partner v003 propose sequentially before revision 1"
    text = knowledge.prompt("Task:\nWrite the memo.\n", sections)
    assert "<<" not in text and text.count(knowledge.NO_OBSERVATION) == 2
    assert ("(The current row produces Artifact B, and the other artifacts are produced by other evaluated communication "
            "flows)<br><br>B \\| wins<br>clearly") in text
    assert "| the worker creates the initial artifact; before revision 1, the Direct manager reviews the current artifact, " \
           "then the Cross-functional partner proposes improvements |" in text
    assert knowledge.lesson(3, REPLY).startswith("## `task_003` — Write the memo\n\n- Domain: Logistics\n")
    for leak in ("see candidate_7", "group-2", "a" * 64):
        try:
            knowledge.check({**REPLY, "critical_feature_analysis": leak})
            raise AssertionError("internal identifiers must be rejected")
        except ValueError:
            pass


# ---- one task end to end -----------------------------------------------------

class ScriptedRuntime:
    """Stakeholders hand off advice; the worker's memo lists the contributions it received."""

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
            return ActionResult("ok", text=f"# Handoff Summary\n\n{action.display_name} advice")
        memo = sandbox / "current_artifact" / "memo.md"
        lines = [f"{c.role} {c.action}" for c in action.incoming] if action.action == "revise" else ["draft"]
        memo.write_text((memo.read_text() if memo.exists() else "") + "".join(f"{line}\n" for line in lines))
        return ActionResult("ok", artifact=sandbox / "current_artifact")


class FakeDomain(Domain):
    name = "fake"

    def __init__(self, data):
        self.data = data

    def load_task(self, task_id):
        return TaskPacket(task_id, f"Write memo {task_id}.", (self.data / "reference_files" / "brief.pdf",), ("memo.md",))


def fake_judge(group, group_dir, judge):
    """Ranks by memo length (more contributions first); equal lengths tie."""
    files = group_dir / "judge_visible" / "artifacts"
    size = {label: len((files / label / "files" / "memo.md").read_text().splitlines()) for label in group.presentation}
    levels = sorted(set(size.values()), reverse=True)
    return {"group_id": group.group_id, "ranking_tiers": [sorted(l for l in size if size[l] == v) for v in levels],
            "hard_invalid": [], "decisive_difference": "Artifact A lists more advice."}


@contextlib.contextmanager
def _fakes():
    calls = []

    def complete_json(profile, *, system, prompt, schema, temperature=None, log_path=None, attempts=3, validate=None):
        if system == planner.SYSTEM["first_round"]:
            aliases = re.findall(r"### `(graph_\d{3})`", prompt)[:5]
            reply = {"task_analysis": ["memo"], "recommendations": [{"structure_id": a, "reason": "fits"} for a in aliases]}
        else:
            assert system == knowledge.SYSTEM and "## Most Effective Communication Edits" in prompt
            reply = REPLY
        validate(reply)
        calls.append(SimpleNamespace(system=system, prompt=prompt, temperature=temperature))
        return reply

    judged = []

    def judge(group, group_dir, settings):
        judged.append(group.group_id)
        return fake_judge(group, group_dir, settings)

    originals = llm.complete_json, ranking.judge_group
    llm.complete_json, ranking.judge_group = complete_json, judge
    try:
        yield calls, judged
    finally:
        llm.complete_json, ranking.judge_group = originals


def test_learning_end_to_end_and_resume():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "data/reference_files").mkdir(parents=True)
        (tmp / "data/reference_files/brief.pdf").write_bytes(b"%PDF brief")
        runtime = ScriptedRuntime()

        def submission(domain, task_id, artifact_dir, out_dir):
            shutil.copytree(artifact_dir, out_dir)
            return sorted(p.name for p in out_dir.iterdir())

        hooks = SimpleNamespace(make_runtime=lambda domain, task_id, profile, config: runtime, submission=submission)
        domain, out = FakeDomain(tmp / "data"), tmp / "learn"
        with _fakes() as (calls, judged):
            records = learn.run(domain, ["t1", "t2"], out_dir=out, hooks=hooks, profile=PROFILE, config=CONFIG)
        first = records[0]
        assert [r["task_id"] for r in records] == ["t1", "t2"] and first["search"] == "completed"
        assert first["rankings"] == ["initial", "search_001", "search_002", "search_003", "promotion"]
        assert first["decision"]["status"] == "updated" and len(first["decision"]["selected"]) == 3
        assert [c.temperature for c in calls] == [0.2] * 4 and "- reference_files/brief.pdf" in calls[0].prompt

        task_dir = out / "tasks" / "t1"
        candidates = json.loads((task_dir / "candidates.json").read_text())
        assert len(list((task_dir / "executions").iterdir())) == len({c["fingerprint"] for c in candidates.values()})
        assert len(candidates) <= 5 + 3 * 3 * 3 and runtime.calls["initial_work"] == runtime.calls["revise"]
        visible = next((task_dir / "rankings" / "search_001").iterdir()) / "judge_visible"
        assert (visible / "task.md").read_text().startswith("Task:\nWrite memo t1.")
        assert json.loads((visible / "deliverable_contract.json").read_text())["expected_basenames"] == ["memo.md"]
        assert (visible / "reference_files/brief.pdf").is_file() and len(list((visible / "artifacts").iterdir())) == 4
        assert visible.parent.name in (visible / "INSTRUCTION.md").read_text()
        promotion = json.loads((task_dir / "rankings" / "promotion.json").read_text())
        assert promotion["status"] == "completed" and len(promotion["candidates"]) == 5

        library = Library(out / "library")
        assert [p.name for p in sorted((out / "snapshots").iterdir())] == ["00_seeds", "01_t1", "02_t2"]
        assert library.lessons() == Library(out / "snapshots/02_t2").lessons()
        assert "## `task_001` — Write the memo" in library.lessons() and "## `task_002` — Write the memo" in library.lessons()
        assert "## `task_002`" not in Library(out / "snapshots/01_t1").lessons()
        promoted = [s for s in library.skills() if s.skill_id.startswith("t1_")]
        assert [s.skill_id for s in promoted] == sorted(f"t1_{c}" for c in first["decision"]["selected"]
                                                        if f"t1_{c}" in first["decision"]["promoted"])

        # A finished stream reruns without model calls; an unfinished task is redone from its boundary,
        # reusing its retrieval, executions and group judgments (only compaction is called again).
        executed = sum(runtime.calls.values())
        (out / "tasks/t2/record.json").unlink()
        with _fakes() as (calls, judged):
            again = learn.run(domain, ["t1", "t2"], out_dir=out, hooks=hooks, profile=PROFILE, config=CONFIG)
        assert again == records and len(calls) == 1 and not judged and sum(runtime.calls.values()) == executed
        assert Library(out / "library").lessons() == library.lessons()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
