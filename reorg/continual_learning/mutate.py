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
"""Mutate(B) of Alg. 1 (App. A.2, "Mutation and promotion").

A child differs from a beam skill s = (V, E) by one edit from five families:

    insert_feedback, insert_propose, insert_loop_in
                    add a node with any role, placed sequentially after an anchor
                    a in {INPUT} u V (it takes over a's outgoing edges) or in
                    parallel with a in V (it shares a's parents and children);
    relocate        remove a node, wire its parents to its children, and place it
                    again sequentially or in parallel at another anchor;
    replace_role    give a node another role, keeping its action and edges.

Sampling draws a family by weight (2/2/1/3/1) among the families that have a
legal edit, then one of its legal edits uniformly without replacement, until
enough children with new execution fingerprints exist. Insertions stop at the
node cap |V| = MAX_NODES. Enumeration order, random draws, node naming and
candidate IDs are deterministic, so a seed fixes the children.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, replace

from ..reflection_skill.graph import (ACTIONS, INPUT, MAX_NODES, OUTPUT, ROLES, RUN_ROLE_IDS, Node,
                                      ReflectionSkill)

FAMILIES = ("insert_feedback", "insert_propose", "insert_loop_in", "relocate", "replace_role")
WEIGHTS = {"insert_feedback": 2, "insert_propose": 2, "insert_loop_in": 1, "relocate": 3, "replace_role": 1}
# Roles sorted by their model-facing IDs (RUN_ROLE_IDS); this order fixes the edits a mutation
# seed draws.
ROLE_ORDER = tuple(sorted(ROLES, key=RUN_ROLE_IDS.get))


@dataclass(frozen=True)
class Edit:
    family: str
    node: str | None = None                          # the relocated or re-roled node
    action: str | None = None                        # the inserted action
    role: str | None = None                          # the inserted or replacement role
    placement: str | None = None                     # "sequential" | "parallel"
    anchor: str | None = None


def legal_edits(skill: ReflectionSkill) -> dict[str, list[Edit]]:
    """Every legal edit of `skill`, by family."""
    ids = [node.node_id for node in skill.nodes]
    edits: dict[str, list[Edit]] = {family: [] for family in FAMILIES}
    if len(ids) < MAX_NODES:
        for action in sorted(ACTIONS):
            family = f"insert_{action}"
            for role in ROLE_ORDER:
                edits[family] += [Edit(family, None, action, role, "sequential", anchor) for anchor in [INPUT, *ids]]
                edits[family] += [Edit(family, None, action, role, "parallel", anchor) for anchor in ids]
    for node in skill.nodes:
        edits["replace_role"] += [Edit("replace_role", node.node_id, role=role) for role in ROLE_ORDER if role != node.role]
        others = [anchor for anchor in ids if anchor != node.node_id]
        edits["relocate"] += [Edit("relocate", node.node_id, placement="sequential", anchor=anchor)
                              for anchor in [INPUT, *others]]
        edits["relocate"] += [Edit("relocate", node.node_id, placement="parallel", anchor=anchor) for anchor in others]
    return edits


def draws(skill: ReflectionSkill, seed: int, weights: dict[str, float] = WEIGHTS):
    """Legal edits in sampled order: a weighted family, then one of its edits uniformly without replacement."""
    rng = random.Random(seed)
    remaining = legal_edits(skill)
    available = [family for family in FAMILIES if remaining[family]]
    while available:
        family = rng.choices(available, weights=[weights[name] for name in available])[0]
        bucket = remaining[family]
        yield bucket.pop(rng.randrange(len(bucket)))
        if not bucket:
            available.remove(family)


def apply(skill: ReflectionSkill, edit: Edit, skill_id: str) -> ReflectionSkill:
    nodes, edges = list(skill.nodes), list(skill.edges)
    if edit.family == "replace_role":
        nodes = [replace(node, role=edit.role) if node.node_id == edit.node else node for node in nodes]
    else:
        if edit.family == "relocate":
            placed = skill.node(edit.node)
            parents = [source for source, target in edges if target == edit.node]
            children = [target for source, target in edges if source == edit.node]
            nodes.remove(placed)
            nodes.append(placed)
            edges = [edge for edge in edges if edit.node not in edge]
            edges += [edge for edge in itertools.product(parents, children) if edge not in edges]
        else:
            base, taken = f"{edit.role}_{edit.action}", {node.node_id for node in nodes}
            placed = Node(next(f"{base}_{k}" for k in itertools.count(1) if f"{base}_{k}" not in taken),
                          edit.role, edit.action)
            nodes.append(placed)
        children = [target for source, target in edges if source == edit.anchor]
        if edit.placement == "sequential":
            edges = [edge for edge in edges if edge[0] != edit.anchor] + [(edit.anchor, placed.node_id)]
        else:
            edges += [(source, placed.node_id) for source, target in edges if target == edit.anchor]
        edges += [(placed.node_id, child) for child in children]
    # Relocation can reconnect INPUT to OUTPUT; that edge routes nothing once the skill has a node.
    edges = [edge for edge in edges if edge != (INPUT, OUTPUT)] if nodes else edges
    return ReflectionSkill(skill_id, tuple(nodes), tuple(edges)).validate()


def children(parent: ReflectionSkill, parent_id: str, round_index: int, *, limit: int, seen: set[str],
             seed: int = 0, weights: dict[str, float] = WEIGHTS) -> list[tuple[str, ReflectionSkill, Edit]]:
    """Up to `limit` children (candidate ID, skill, edit) whose fingerprints are not in `seen`, which
    is updated. Candidate IDs count every drawn edit, including duplicates."""
    found: list[tuple[str, ReflectionSkill, Edit]] = []
    draw_seed = seed + round_index * 1009 + sum(parent_id.encode("utf-8"))
    for index, edit in enumerate(draws(parent, draw_seed, weights), start=1):
        if len(found) == limit:
            break
        candidate_id = f"r{round_index:03d}_{parent_id}_c{index:03d}"
        child = apply(parent, edit, candidate_id)
        fingerprint = child.fingerprint()
        if fingerprint not in seen:
            seen.add(fingerprint)
            found.append((candidate_id, child, edit))
    return found
