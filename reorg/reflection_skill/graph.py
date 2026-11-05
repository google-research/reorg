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
"""Organizational reflection skills (Sec. 3.3, App. A.1).

A one-round skill is a DAG s = (V, E). Each node v pairs a stakeholder role
rho_v with an action a_v in {Feedback, Propose, Loop-In}; an edge (v_i, v_j)
routes v_i's contribution and context to v_j. Two markers bound the graph:
INPUT is the current artifact A_{r-1}, OUTPUT is the worker's revision A_r.
Every node lies on a path INPUT -> ... -> OUTPUT. The empty skill is the
single edge INPUT -> OUTPUT: no stakeholder calls, just a worker revision.

A skill stores roles, actions and structure only. Role profiles and action
prompts are fixed files under reorg/prompts/, so the same skill runs on any task.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass

from .. import prompts

INPUT, OUTPUT = "INPUT", "OUTPUT"
ACTIONS = ("feedback", "propose", "loop_in")
MAX_NODES = 10                                   # stakeholder nodes per reflection round

# Role IDs (App. A.7) and the display names that the prompts and Loop-In show the models.
ROLES = {
    "same_job_teammate": "Same-job teammate",
    "direct_manager": "Direct manager",
    "upstream_requester": "Upstream requester",
    "cross_functional_partner": "Cross-functional partner",
    "technical_customer": "Technical customer",
    "implementation_partner": "Implementation partner",
    "procurement_evaluator": "Procurement evaluator",
    "legal_risk_reviewer": "Legal and risk reviewer",
    "internal_auditor": "Internal auditor",
    "experienced_mentor": "Experienced mentor",
}
WORKER_DISPLAY_NAME = "Task Owner"

# Role IDs used in model-facing text (e.g. contribution headings
# "## direct_manager_v003 / feedback / available"); code uses the keys of ROLES.
RUN_ROLE_IDS = {**{role: f"{role}_v003" for role in ROLES},
                "cross_functional_partner": "xfn_partner_v003", "technical_customer": "customer_technical_v003"}

# The action prompt behind each action; Loop-In also runs its three protocol prompts.
ACTION_PROMPTS = {"feedback": "actions/feedback.md", "propose": "actions/propose.md", "loop_in": "loop_in/loop_in.md"}
LOOP_IN_PROMPTS = ("loop_in/select.md", "loop_in/discussion.md", "loop_in/summary.md")


@dataclass(frozen=True)
class Node:
    node_id: str
    role: str
    action: str

    @property
    def display_name(self) -> str:
        return ROLES[self.role]


@dataclass(frozen=True)
class ReflectionSkill:
    skill_id: str
    nodes: tuple[Node, ...]
    edges: tuple[tuple[str, str], ...]

    # ---- structure ---------------------------------------------------------
    def node(self, node_id: str) -> Node:
        return next(node for node in self.nodes if node.node_id == node_id)

    def parents(self, node_id: str) -> tuple[str, ...]:
        """Stakeholder parents of a node, or of OUTPUT (the nodes the worker hears from directly)."""
        return tuple(sorted(source for source, target in self.edges if target == node_id and source != INPUT))

    def frontiers(self) -> list[list[Node]]:
        """Stakeholder nodes in topological layers; a layer runs once all its parents finish."""
        indegree = {node.node_id: 0 for node in self.nodes}
        children: dict[str, list[str]] = defaultdict(list)
        for source, target in self.edges:
            if source != INPUT and target != OUTPUT:
                indegree[target] += 1
                children[source].append(target)
        ready, layers = sorted(node_id for node_id, degree in indegree.items() if degree == 0), []
        while ready:
            layers.append([self.node(node_id) for node_id in ready])
            following = []
            for node_id in ready:
                for child in children[node_id]:
                    indegree[child] -= 1
                    if indegree[child] == 0:
                        following.append(child)
            ready = sorted(following)
        if sum(map(len, layers)) != len(self.nodes):
            raise ValueError(f"{self.skill_id}: graph has a cycle")
        return layers

    def validate(self) -> "ReflectionSkill":
        ids = [node.node_id for node in self.nodes]
        if len(set(ids)) != len(ids) or {INPUT, OUTPUT} & set(ids):
            raise ValueError(f"{self.skill_id}: node IDs must be unique and differ from the markers")
        if len(self.nodes) > MAX_NODES:
            raise ValueError(f"{self.skill_id}: {len(self.nodes)} nodes exceed the cap of {MAX_NODES}")
        for node in self.nodes:
            if node.role not in ROLES or node.action not in ACTIONS:
                raise ValueError(f"{self.skill_id}: unknown role or action in {node}")
        known = set(ids) | {INPUT, OUTPUT}
        if len(set(self.edges)) != len(self.edges):
            raise ValueError(f"{self.skill_id}: duplicate edge")
        for source, target in self.edges:
            if source not in known or target not in known or source == target or source == OUTPUT or target == INPUT:
                raise ValueError(f"{self.skill_id}: invalid edge {source} -> {target}")
        if not self.nodes and self.edges != ((INPUT, OUTPUT),):
            raise ValueError(f"{self.skill_id}: the empty skill is exactly INPUT -> OUTPUT")
        if self.nodes and (INPUT, OUTPUT) in self.edges:
            raise ValueError(f"{self.skill_id}: INPUT -> OUTPUT is only allowed in the empty skill")
        self.frontiers()
        reach_from_input, reach_to_output = self._reachable(INPUT, forward=True), self._reachable(OUTPUT, forward=False)
        stranded = sorted(node_id for node_id in ids if node_id not in reach_from_input or node_id not in reach_to_output)
        if stranded:
            raise ValueError(f"{self.skill_id}: nodes off every INPUT -> OUTPUT path: {stranded}")
        return self

    def _reachable(self, start: str, *, forward: bool) -> set[str]:
        step: dict[str, list[str]] = defaultdict(list)
        for source, target in self.edges:
            step[source if forward else target].append(target if forward else source)
        seen, pending = set(), [start]
        while pending:
            for following in step[pending.pop()]:
                if following not in seen:
                    seen.add(following)
                    pending.append(following)
        return seen

    # ---- identity ----------------------------------------------------------
    def fingerprint(self) -> str:
        """Execution identity, invariant to node IDs and edge order: Weisfeiler-Lehman
        colors over (role, action, prompt hashes), refined |V| times. Two skills with
        the same fingerprint make the same model calls in the same structure."""
        labels: dict[str, dict] = {INPUT: {"marker": INPUT}, OUTPUT: {"marker": OUTPUT}}
        for node in self.nodes:
            labels[node.node_id] = {
                "role": node.role,
                "action": node.action,
                "prompts": [_sha(f"roles/{node.role}.md"), _sha(ACTION_PROMPTS[node.action])]
                + ([_sha(name) for name in LOOP_IN_PROMPTS] if node.action == "loop_in" else []),
            }
        parents: dict[str, list[str]] = {key: [] for key in labels}
        children: dict[str, list[str]] = {key: [] for key in labels}
        for source, target in self.edges:
            parents[target].append(source)
            children[source].append(target)
        colors = {key: _digest(label) for key, label in labels.items()}
        for _ in range(len(labels)):
            colors = {key: _digest({"label": labels[key],
                                    "parents": sorted(colors[p] for p in parents[key]),
                                    "children": sorted(colors[c] for c in children[key])})
                      for key in labels}
        return _digest({"version": "reorg_skill.v1",
                        "nodes": sorted(colors.values()),
                        "edges": sorted([colors[s], colors[t]] for s, t in self.edges)})

    # ---- serialization -----------------------------------------------------
    def to_dict(self) -> dict:
        return {"skill_id": self.skill_id,
                "nodes": [{"node_id": n.node_id, "role": n.role, "action": n.action} for n in self.nodes],
                "edges": [list(edge) for edge in self.edges]}

    @classmethod
    def from_dict(cls, payload: dict) -> "ReflectionSkill":
        return cls(skill_id=str(payload["skill_id"]),
                   nodes=tuple(Node(**row) for row in payload["nodes"]),
                   edges=tuple((str(s), str(t)) for s, t in payload["edges"])).validate()


def empty_skill(skill_id: str = "empty") -> ReflectionSkill:
    return ReflectionSkill(skill_id, (), ((INPUT, OUTPUT),))


def _sha(prompt_name: str) -> str:
    return prompts.manifest()[prompt_name]["sha256"]


def _digest(payload) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
