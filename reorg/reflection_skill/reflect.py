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
"""One reflection round (Eq. 1):

    F_r = Reflect(x, A_{r-1}, H_{r-1}; s_r)        the stakeholder nodes of s_r
    A_r = W(x, A_{r-1}, H_{r-1}, F_r)              the worker's revision

Stakeholder nodes run in topological layers, in parallel within a layer, each
in its own sandbox holding copies of the task, the references and the current
artifact. A node receives the contributions routed to it: its parents' and,
transitively, theirs. Contributions and chronicle updates are published in
node-ID order once a layer finishes. A failed stakeholder publishes a
"missing" contribution and the round continues; only a failed worker leaves
the artifact unchanged.

A domain plugs in through `Runtime`: it stages a sandbox and runs one action.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

from . import chronicle as chron
from .graph import OUTPUT, ROLES, WORKER_DISPLAY_NAME, ReflectionSkill

CONTRIBUTIONS = "collaboration_workspace/contributions"
CONTRIBUTION_INDEX = "collaboration_workspace/colleague_contributions.md"
DEPLOYABLE = ("ok", "turn_limit_best_available")    # worker outcomes whose artifact is kept


@dataclass(frozen=True)
class Contribution:
    contribution_id: str                            # content hash, so routing order is reproducible
    vertex_id: str
    role: str
    action: str
    status: str                                     # "available" | "missing"
    text: str
    upstream: tuple[str, ...] = ()                  # contributions this one was written from

    @classmethod
    def create(cls, vertex_id: str, role: str, action: str, text: str, status: str,
               upstream: tuple[str, ...]) -> "Contribution":
        text = text.strip()
        identity = json.dumps({"vertex_id": vertex_id, "action": action, "role_id": role, "status": status,
                               "text": text, "upstream": list(upstream)},
                              sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        return cls(f"contribution_{digest[:24]}", vertex_id, role, action, status, text, upstream)


@dataclass(frozen=True)
class Action:
    vertex_id: str
    action: str                                     # initial_work | revise | feedback | propose | loop_in
    role: str | None                                # None for the worker
    incoming: tuple[Contribution, ...] = ()

    @property
    def display_name(self) -> str:
        return ROLES[self.role] if self.role else WORKER_DISPLAY_NAME


@dataclass
class ActionResult:
    status: str                                     # ok | turn_limit_best_available | model_failure |
                                                    # protocol_failure | infrastructure_error
    text: str = ""                                  # a stakeholder's published contribution
    artifact: Path | None = None                    # the worker's artifact directory
    logs: dict = field(default_factory=dict)
    usage: dict | None = None


class Runtime(Protocol):
    """Domain-owned execution (data/<name>/runtime.py)."""

    # What a missing contribution tells the worker to inspect; LAB uses "deliverables".
    inspect_subject: str = "artifact"

    def stage(self, sandbox: Path, artifact: Path | None) -> None:
        """Write task.md, reference_files/, current_artifact/ (a copy of `artifact`) and scratch/."""

    def run(self, action: Action, sandbox: Path, logs: Path) -> ActionResult:
        """Run one action in a staged sandbox. Raising means an infrastructure failure."""


@dataclass
class RoundResult:
    status: str                                     # strict | degraded | incomplete
    artifact: Path                                  # A_r, or A_{r-1} if the worker failed
    contributions: list[Contribution]
    records: list[dict]

    def to_dict(self) -> dict:
        return {"status": self.status, "artifact": str(self.artifact),
                "contributions": [asdict(c) for c in self.contributions], "records": self.records}


def route(skill: ReflectionSkill, node_id: str, published: dict[str, Contribution]) -> tuple[Contribution, ...]:
    """Contributions routed to a node (or to OUTPUT): its parents' and, transitively, theirs."""
    by_id = {c.contribution_id: c for c in published.values()}
    routed = {published[p].contribution_id: published[p] for p in skill.parents(node_id) if p in published}
    pending = [upstream for c in routed.values() for upstream in c.upstream]
    while pending:
        contribution = by_id.get(pending.pop())
        if contribution and contribution.contribution_id not in routed:
            routed[contribution.contribution_id] = contribution
            pending.extend(contribution.upstream)
    return tuple(routed[key] for key in sorted(routed))


def stage_contributions(sandbox: Path, incoming: tuple[Contribution, ...]) -> None:
    """collaboration_workspace/contributions/<id>.md plus an index the prompts point to."""
    (sandbox / CONTRIBUTIONS).mkdir(parents=True, exist_ok=True)
    lines = ["# Colleague Contributions", ""]
    for contribution in incoming:
        path = f"{CONTRIBUTIONS}/{contribution.contribution_id}.md"
        (sandbox / path).write_text(contribution.text + "\n", encoding="utf-8", newline="\n")
        lines.append(f"- `{path}` status={contribution.status}")
    (sandbox / CONTRIBUTION_INDEX).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def run_action(action: Action, runtime: Runtime, artifact: Path | None, chronicle: chron.WorkChronicle,
               node_dir: Path, retries: int) -> tuple[ActionResult, Path, str]:
    """Stage and run one action, retrying infrastructure failures. Returns (result, sandbox, chronicle base)."""
    for attempt in range(1, retries + 2):
        attempt_dir = node_dir / f"attempt_{attempt:02d}"
        shutil.rmtree(attempt_dir, ignore_errors=True)
        sandbox = attempt_dir / "sandbox"
        runtime.stage(sandbox, artifact)
        stage_contributions(sandbox, action.incoming)
        base = chronicle.stage(sandbox)
        try:
            result = runtime.run(action, sandbox, attempt_dir / "logs")
        except Exception as exc:
            result = ActionResult("infrastructure_error", logs={"error": repr(exc)})
        if result.status != "infrastructure_error":
            break
    return result, sandbox, base


def work(action: Action, runtime: Runtime, artifact: Path | None, chronicle: chron.WorkChronicle,
         out_dir: Path, retries: int = 1) -> ActionResult:
    """A worker action (initial work or revision). A usable artifact is copied to out_dir/artifact;
    otherwise result.artifact is None."""
    result, sandbox, base = run_action(action, runtime, artifact, chronicle, out_dir / "nodes" / action.vertex_id, retries)
    produced = result.artifact
    if result.status in DEPLOYABLE and produced and any(p.is_file() and p.stat().st_size for p in produced.rglob("*")):
        result.artifact = out_dir / "artifact"
        shutil.rmtree(result.artifact, ignore_errors=True)
        shutil.copytree(produced, result.artifact)
        chronicle.merge(sandbox, base, chron.label(action.display_name, action.action, action.vertex_id),
                        fallback=f"Typed action {action.vertex_id} completed with status {result.status}.")
    else:
        result.artifact = None
    return result


def _publish(action: Action, result: ActionResult, sandbox: Path, base: str,
             chronicle: chron.WorkChronicle, subject: str) -> Contribution:
    upstream = tuple(c.contribution_id for c in action.incoming)
    label = chron.label(action.display_name, action.action, action.vertex_id)
    # A stakeholder that hit its turn limit after writing a valid handoff still counts.
    if result.status in DEPLOYABLE and "# Handoff Summary" in result.text:
        chronicle.merge(sandbox, base, label, fallback=chron.handoff_summary(result.text))
        return Contribution.create(action.vertex_id, action.role, action.action, result.text, "available", upstream)
    status = result.status if result.status != "ok" else "invalid_contribution"
    chronicle.append(label, f"Runtime status: {status}. No model-authored work chronicle update was accepted.")
    text = (f"# Handoff Summary\n\nmissing_contribution: status={status}. "
            f"The task owner must independently inspect the {subject}.")
    return Contribution.create(action.vertex_id, action.role, action.action, text, "missing", upstream)


def reflect(skill: ReflectionSkill, runtime: Runtime, artifact: Path, chronicle: chron.WorkChronicle,
            round_dir: Path, parallelism: int = 1, retries: int = 1) -> RoundResult:
    """Run s_r's stakeholders on A_{r-1}, then the worker's revision."""
    published: dict[str, Contribution] = {}
    records: list[dict] = []
    for layer in skill.frontiers():
        actions = [Action(n.node_id, n.action, n.role, route(skill, n.node_id, published)) for n in layer]
        with ThreadPoolExecutor(max_workers=max(1, min(parallelism, len(actions)))) as pool:
            outcomes = list(pool.map(lambda a: run_action(a, runtime, artifact, chronicle,
                                                          round_dir / "nodes" / a.vertex_id, retries), actions))
        for action, (result, sandbox, base) in zip(actions, outcomes):     # publish in node-ID order
            contribution = _publish(action, result, sandbox, base, chronicle,
                                    getattr(runtime, "inspect_subject", "artifact"))
            published[action.vertex_id] = contribution
            records.append({"vertex_id": action.vertex_id, "role": action.role, "action": action.action,
                            "status": result.status, "contribution": contribution.contribution_id,
                            "logs": result.logs, "usage": result.usage})

    revise = Action("worker_revise", "revise", None, route(skill, OUTPUT, published))
    result = work(revise, runtime, artifact, chronicle, round_dir, retries)
    records.append({"vertex_id": revise.vertex_id, "role": None, "action": "revise", "status": result.status,
                    "logs": result.logs, "usage": result.usage})
    contributions = [published[key] for key in sorted(published)]
    if result.artifact is None:
        return RoundResult("incomplete", artifact, contributions, records)
    strict = result.status == "ok" and all(c.status == "available" for c in contributions)
    return RoundResult("strict" if strict else "degraded", result.artifact, contributions, records)
