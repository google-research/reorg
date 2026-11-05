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
"""Reflection core with a scripted runtime: skills, routing, chronicle, rounds, Loop-In. No API calls."""

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reorg.reflection_skill import chronicle as chron  # noqa: E402
from reorg.reflection_skill import loop_in  # noqa: E402
from reorg.reflection_skill.reflect import ActionResult, reflect, route, work, Action  # noqa: E402
from reorg.reflection_skill.graph import INPUT, OUTPUT, Node, ReflectionSkill, empty_skill  # noqa: E402


def _skill(skill_id="s", nodes=(("a", "direct_manager", "feedback"), ("b", "legal_risk_reviewer", "propose")),
           edges=((INPUT, "a"), ("a", "b"), ("b", OUTPUT))):
    return ReflectionSkill(skill_id, tuple(Node(*n) for n in nodes), tuple(edges)).validate()


def _rejects(**kwargs):
    try:
        _skill(**kwargs)
    except ValueError:
        return True
    return False


# ---- skills ----------------------------------------------------------------

def test_skill_validation():
    empty_skill().validate()
    assert _rejects(edges=((INPUT, "a"), ("a", OUTPUT)))                       # b stranded
    assert _rejects(edges=((INPUT, "a"), ("a", "b"), ("b", "a"), ("b", OUTPUT)))  # cycle
    assert _rejects(nodes=(("a", "ceo", "feedback"),), edges=((INPUT, "a"), ("a", OUTPUT)))
    assert _rejects(nodes=(("a", "direct_manager", "verify"),), edges=((INPUT, "a"), ("a", OUTPUT)))
    assert _rejects(nodes=(), edges=((INPUT, OUTPUT), (INPUT, OUTPUT)))
    many = tuple((f"n{i}", "direct_manager", "feedback") for i in range(11))
    assert _rejects(nodes=many, edges=tuple((INPUT, f"n{i}") for i in range(11)) + tuple((f"n{i}", OUTPUT) for i in range(11)))


def test_fingerprint_ignores_ids_and_edge_order_but_not_structure():
    base = _skill()
    renamed = _skill(nodes=(("x", "direct_manager", "feedback"), ("y", "legal_risk_reviewer", "propose")),
                     edges=(("y", OUTPUT), ("x", "y"), (INPUT, "x")))
    parallel = _skill(edges=((INPUT, "a"), (INPUT, "b"), ("a", OUTPUT), ("b", OUTPUT)))
    swapped = _skill(nodes=(("a", "direct_manager", "propose"), ("b", "legal_risk_reviewer", "feedback")))
    assert base.fingerprint() == renamed.fingerprint()
    assert len({base.fingerprint(), parallel.fingerprint(), swapped.fingerprint()}) == 3
    assert ReflectionSkill.from_dict(base.to_dict()).fingerprint() == base.fingerprint()


def test_frontiers_and_routing_are_transitive():
    skill = _skill(nodes=(("a", "direct_manager", "feedback"), ("b", "internal_auditor", "feedback"),
                          ("c", "legal_risk_reviewer", "propose")),
                   edges=((INPUT, "a"), (INPUT, "b"), ("a", "c"), ("b", OUTPUT), ("c", OUTPUT)))
    assert [[n.node_id for n in layer] for layer in skill.frontiers()] == [["a", "b"], ["c"]]
    from reorg.reflection_skill.reflect import Contribution
    a = Contribution.create(vertex_id="a", role="direct_manager", action="feedback", text="A", status="available", upstream=())
    b = Contribution.create(vertex_id="b", role="internal_auditor", action="feedback", text="B", status="available", upstream=())
    c = Contribution.create(vertex_id="c", role="legal_risk_reviewer", action="propose", text="C", status="available",
                            upstream=(a.contribution_id,))
    published = {"a": a, "b": b, "c": c}
    assert {x.vertex_id for x in route(skill, "c", published)} == {"a"}
    assert {x.vertex_id for x in route(skill, OUTPUT, published)} == {"a", "b", "c"}


# ---- chronicle -------------------------------------------------------------

def test_chronicle_update_extraction():
    assert chron.extract_update("\n## Work Chronicle Update\n\nDid X.\n") == "Did X."
    assert chron.extract_update("## Work Chronicle Update - round 1\nDid X.") == "Did X."
    assert chron.extract_update("## Manager\n### Work Chronicle Update\nDid X.") == "Did X."
    assert chron.extract_update("") == chron.extract_update("random notes") == ""
    assert chron.extract_update("## Work Chronicle Update\na\n## Work Chronicle Update\nb") == ""
    assert chron.handoff_summary("# Handoff Summary\n\nFix the totals.\n\n# Details\nmore") == "Fix the totals."


# ---- rounds ----------------------------------------------------------------

class ScriptedRuntime:
    """Stakeholders publish a handoff naming their inputs; the worker appends a line to its artifact."""

    def __init__(self, fail=()):
        self.fail, self.seen = set(fail), {}

    def stage(self, sandbox, artifact):
        (sandbox / "current_artifact").mkdir(parents=True, exist_ok=True)
        if artifact is not None:
            shutil.copytree(artifact, sandbox / "current_artifact", dirs_exist_ok=True)

    def run(self, action, sandbox, logs):
        self.seen[action.vertex_id] = sorted(c.vertex_id for c in action.incoming)
        if action.vertex_id in self.fail:
            return ActionResult("model_failure")
        staged = sandbox / chron.STAGED
        staged.write_text(staged.read_text() + f"\n## Work Chronicle Update\n\n{action.vertex_id} done.\n")
        if action.role:
            return ActionResult("ok", text=f"# Handoff Summary\n\n{action.vertex_id} advice")
        memo = sandbox / "current_artifact" / "memo.md"
        memo.write_text((memo.read_text() if memo.exists() else "") + f"{action.action}\n")
        return ActionResult("ok", artifact=sandbox / "current_artifact")


def _round(runtime, skill):
    tmp = Path(tempfile.mkdtemp())
    chronicle = chron.WorkChronicle(tmp / "work_chronicle.md")
    first = work(Action("worker_initial", "initial_work", None), runtime, None, chronicle, tmp / "a0")
    return reflect(skill, runtime, first.artifact, chronicle, tmp / "r1", parallelism=2), chronicle, tmp


def test_round_routes_contributions_and_revises():
    runtime = ScriptedRuntime()
    result, chronicle, tmp = _round(runtime, _skill())
    assert result.status == "strict" and (result.artifact / "memo.md").read_text() == "initial_work\nrevise\n"
    assert runtime.seen == {"worker_initial": [], "a": [], "b": ["a"], "worker_revise": ["a", "b"]}
    text = chronicle.text()
    order = [text.index(f"- {label} -") for label in ("First Iter Work", "Feedback", "Propose", "Improve Work")]
    assert order == sorted(order) and "No action updates yet." not in text
    shutil.rmtree(tmp)


def test_failed_stakeholder_degrades_and_failed_worker_keeps_artifact():
    result, chronicle, tmp = _round(ScriptedRuntime(fail={"a"}), _skill())
    assert result.status == "degraded" and result.contributions[0].status == "missing"
    assert "Runtime status: model_failure" in chronicle.text()
    shutil.rmtree(tmp)
    result, _, tmp = _round(ScriptedRuntime(fail={"worker_revise"}), _skill())
    assert result.status == "incomplete" and (result.artifact / "memo.md").read_text() == "initial_work\n"
    shutil.rmtree(tmp)


def test_empty_skill_is_a_plain_revision():
    runtime = ScriptedRuntime()
    result, _, tmp = _round(runtime, empty_skill())
    assert result.status == "strict" and runtime.seen["worker_revise"] == [] and not result.contributions
    shutil.rmtree(tmp)


# ---- Loop-In ---------------------------------------------------------------

class ScriptedDriver:
    def __init__(self, partner="Internal auditor"):
        self.partner, self.prompts = partner, []

    def select(self, initiator, protocol_prompt):
        self.prompts.append(protocol_prompt)
        return self.partner, "Can you check the reserve math?"

    def turn(self, speaker, protocol_prompt, label):
        self.prompts.append(protocol_prompt)
        return f"{speaker} says hi"

    def publish(self, initiator, protocol_prompt):
        self.prompts.append(protocol_prompt)
        return "# Handoff Summary\n\nExtend the reserve."


def test_loop_in_protocol():
    driver = ScriptedDriver()
    result = loop_in.run("direct_manager", driver)
    assert result.status == "ok" and result.partner == "internal_auditor" and len(result.messages) == 3
    assert "- Direct manager" not in driver.prompts[0] and "- Internal auditor" in driver.prompts[0]
    assert "Message 1 - Direct manager:\nCan you check the reserve math?" in driver.prompts[1]
    text = loop_in.outcome(result)
    assert text.startswith("# Handoff Summary") and "## Opening - Direct manager" in text and "## Round 1 - Internal auditor" in text
    bad = loop_in.run("direct_manager", ScriptedDriver(partner="Direct manager"))
    assert bad.status == "partial" and bad.phase == "select"


def test_protocol_templates_need_exact_variables():
    try:
        loop_in.render("loop_in/summary.md", {"transcript": "t", "extra": 1})
        raise AssertionError("extra variables must be rejected")
    except ValueError:
        pass


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
