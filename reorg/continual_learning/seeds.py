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
"""The initial library L_0 (Alg. 1; App. A.2, "Library Learning and Artifact Ranking").

Five seeds: the empty skill, the direct manager's Feedback, Propose and
Loop-In, and the manager's Feedback followed by Propose. Skill and node IDs
matter: the first-round planner numbers skills in skill-ID order, and
mutations name new nodes after existing ones.
"""

from __future__ import annotations

from ..reflection_skill.graph import INPUT, OUTPUT, Node, ReflectionSkill, empty_skill


def _manager(skill_id: str, *actions: str) -> ReflectionSkill:
    """The direct manager's actions in sequence between INPUT and OUTPUT."""
    ids = [f"direct_manager_{action}_1" for action in actions]
    nodes = tuple(Node(node_id, "direct_manager", action) for node_id, action in zip(ids, actions))
    return ReflectionSkill(skill_id, nodes, tuple(zip([INPUT, *ids], [*ids, OUTPUT]))).validate()


SEEDS = (
    empty_skill("initial_001_single_round_empty"),
    _manager("initial_002_single_round_manager_feedback", "feedback"),
    _manager("initial_003_single_round_manager_propose", "propose"),
    _manager("initial_004_single_round_manager_loop_in", "loop_in"),
    _manager("initial_005_single_round_manager_feedback_propose", "feedback", "propose"),
)
