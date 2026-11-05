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
"""The planner (Sec. 3.2; App. A.3, "Deployment").

Before reflection round r the planner selects a skill s_r from the frozen
library L using the planner knowledge M, or stops:

    r = 1   first-round retrieval (P5): rank the top 5 skills of L for the task x,
            given the lessons in M; the caller runs the first.
    r > 1   work-state projection (P8): summarize x and the chronicle H_{r-1} into
            the requested work, attributed suggestions and completed work;
            completion gate (P9): score that state from 1 to 10, stop at >= 9;
            later-round retrieval (P10): rank all unused nonempty skills and
            leave_unchanged given the state and the precedents in M; the model
            returns 9 skills plus leave_unchanged.

The gate alone stops: the controller runs the highest-ranked skill even when
leave_unchanged ranks first. Skills reach the model under aliases (graph_###,
flow_###) as plain-language flows built from role display names, never role
IDs.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from . import llm, prompts
from .domain import TaskPacket
from .library import Library
from .reflection_skill.graph import ROLES, ReflectionSkill

STOP = "leave_unchanged"
ACTION_PHRASES = {"feedback": "reviews the current artifact",
                  "propose": "proposes improvements",
                  "loop_in": "consults another colleague and returns a joint recommendation"}
# System prompts, one per planner call.
SYSTEM = {
    "first_round": "Follow the self-contained steering-flow recommendation document. Return only valid JSON.",
    "projection": "Project the current communication state and return only valid JSON.",
    "completion": "Estimate current task completion and return only valid JSON.",
    "later_round": "Recommend and rank the next communication options. Return only valid JSON.",
}
# Ranking uses a label-free judge, so no task carries an absolute "helpful" label.
HELPFUL_TASKS = ("No absolute helpful-task labels are available under the ordinal unsupervised judge. "
                 "Use the task-specific observations below instead.")
_PLACEHOLDER = re.compile(r"<<([A-Z][A-Z0-9_]*)>>")


def render(name: str, values: dict[str, str]) -> str:
    """Fill a <<PLACEHOLDER>> template; every declared placeholder must be given, and only those."""
    source = prompts.load(name)
    declared = set(_PLACEHOLDER.findall(source))
    if declared != set(values):
        raise ValueError(f"{name}: placeholders {sorted(declared)} != given {sorted(values)}")
    return _PLACEHOLDER.sub(lambda match: values[match.group(1)], source)


def task_query(packet: TaskPacket, data_root: Path) -> str:
    """x as the first-round planner reads it: request, reference files, expected deliverables."""
    references = [path.relative_to(data_root).as_posix() for path in packet.references]
    return "\n\n".join([f"Task:\n{packet.request.strip()}",
                        f"Reference files:\n{_bullets(references, 'None')}",
                        f"Expected deliverables:\n{_bullets(packet.deliverables, 'None declared')}"]).strip() + "\n"


def _bullets(values, empty: str) -> str:
    return "\n".join(f"- {value}" for value in values if value) or f"- {empty}"


def communication_flow(skill: ReflectionSkill) -> str:
    """The stakeholder stages of s in words, e.g. "the Direct manager reviews the current
    artifact, then the Internal auditor proposes improvements"; "" for the empty skill."""
    # A stage is a frontier layer: the nodes at one longest-path depth.
    stages = (sorted(f"the {node.display_name} {ACTION_PHRASES[node.action]}" for node in layer)
              for layer in skill.frontiers())
    return ", then ".join(" in parallel with ".join(stage) for stage in stages)


def reviewer_profiles() -> str:
    """One line per role: display name and the second and third paragraphs of its profile."""
    rows = []
    for role, name in ROLES.items():
        paragraphs = [part.strip().replace("\n", " ") for part in prompts.load(f"roles/{role}.md").split("\n\n")
                      if part.strip()]
        rows.append(f"- **{name}**: {' '.join(paragraphs[1:3])}")
    return "\n".join(rows)


def _schema(name: str) -> dict:
    return json.loads(prompts.load(name))


# ---- first round -------------------------------------------------------------

def first_round_prompt(task: str, skills: list[ReflectionSkill], observations: dict[str, int], lessons: str,
                       top_k: int) -> tuple[str, dict[str, str]]:
    """The P5 prompt and its alias map graph_### -> skill ID (skills numbered in skill-ID order)."""
    aliases = {f"graph_{index:03d}": skill.skill_id for index, skill in enumerate(skills, start=1)}
    catalog = []
    for alias, skill in zip(aliases, skills):
        flow = communication_flow(skill)
        catalog += [f"### `{alias}`", "",
                    "- Communication flow: the worker creates the initial artifact; "
                    + (f"before revision 1, {flow}" if flow else "revision 1 proceeds without colleague communication"),
                    f"- Prior top-three promotion or reinforcement observations: {observations.get(skill.skill_id, 0)}",
                    ""]
    prompt = render("planner/first_round_retrieval.md", {
        "TASK": task.strip(), "REVIEWER_PROFILES": reviewer_profiles(),
        "STRUCTURE_CATALOG": "\n".join(catalog).rstrip(), "HELPFUL_TASKS": HELPFUL_TASKS,
        "LEARNED_INSIGHTS": lessons.strip(), "TOP_K": str(top_k)})
    return prompt.rstrip() + "\n", aliases


def first_round_schema(allowed: list[str], top_k: int) -> dict:
    return {"type": "object",
            "properties": {
                "task_analysis": {"type": "array", "items": {"type": "string"}},
                "recommendations": {
                    "type": "array", "minItems": top_k, "maxItems": top_k,
                    "items": {"type": "object",
                              "properties": {"structure_id": {"type": "string", "enum": allowed},
                                             "reason": {"type": "string"}},
                              "required": ["structure_id", "reason"]}}},
            "required": ["task_analysis", "recommendations"]}


def check_first_round(reply: dict, allowed: list[str], top_k: int) -> None:
    rows = reply["recommendations"]
    ids = [row["structure_id"] for row in rows]
    if (set(reply) != {"task_analysis", "recommendations"} or not reply["task_analysis"]
            or not all(_nonblank(item) for item in reply["task_analysis"])
            or any(set(row) != {"structure_id", "reason"} or not _nonblank(row["reason"]) for row in rows)
            or len(ids) != top_k or len(set(ids)) != top_k or not set(ids) <= set(allowed)):
        raise ValueError(f"expected {top_k} unique available graph aliases with reasons and a task analysis")


def select_first_round(profile: llm.ModelProfile, task: str, library: Library, *, top_k: int = 5,
                       temperature: float | None = None, log_path: Path | None = None) -> list[tuple[str, str]]:
    """The top_k skills best-first as (skill ID, reason)."""
    skills = library.skills()
    top_k = min(top_k, len(skills))
    prompt, aliases = first_round_prompt(task, skills, library.observations(), library.lessons(), top_k)
    allowed = list(aliases)
    reply = llm.complete_json(profile, system=SYSTEM["first_round"], prompt=prompt,
                              schema=first_round_schema(allowed, top_k), temperature=temperature, log_path=log_path,
                              validate=lambda value: check_first_round(value, allowed, top_k))
    return [(aliases[row["structure_id"]], row["reason"]) for row in reply["recommendations"]]


# ---- later rounds ------------------------------------------------------------

def projection_prompt(task: str, chronicle: str) -> str:
    return render("planner/state_projection.md", {"TASK": task, "WORK_CHRONICLE": chronicle})


def project_work_state(profile: llm.ModelProfile, task: str, chronicle: str, *, temperature: float | None = None,
                       log_path: Path | None = None) -> dict:
    """The work state {task_summary, what_was_suggested, what_was_done} of (x, H_{r-1})."""
    return llm.complete_json(profile, system=SYSTEM["projection"], prompt=projection_prompt(task, chronicle),
                             schema=_schema("planner/state_projection.schema.json"), temperature=temperature,
                             log_path=log_path)


def state_text(state: dict) -> str:
    suggested = "\n".join(f"- {item}" for item in state["what_was_suggested"]) or "- None recorded."
    done = "\n".join(f"- {item}" for item in state["what_was_done"]) or "- None recorded."
    return (f"## Task and current stage\n\n{state['task_summary']}\n\n"
            f"## Suggestions so far\n\n{suggested}\n\n"
            f"## Work done so far\n\n{done}\n")


def completion_prompt(state: dict) -> str:
    return render("planner/completion_assessment.md",
                  {"CURRENT_WORK_SUMMARY": state_text(state).rstrip()}).rstrip() + "\n"


def completion_gate(profile: llm.ModelProfile, state: dict, *, temperature: float | None = None,
                    log_path: Path | None = None) -> dict:
    """{analysis, task_completion_score}, the score an integer from 1 to 10."""
    def check(reply: dict) -> None:
        if not _nonblank(reply["analysis"]):
            raise ValueError("analysis must be nonblank")
    reply = llm.complete_json(profile, system=SYSTEM["completion"], prompt=completion_prompt(state),
                              schema=_schema("planner/completion_assessment.schema.json"), temperature=temperature,
                              log_path=log_path, validate=check)
    return {"analysis": reply["analysis"].strip(), "task_completion_score": reply["task_completion_score"]}


def later_round_flows(skills: list[ReflectionSkill], used: list[ReflectionSkill]) -> list[tuple[str, ReflectionSkill]]:
    """(flow_###, skill) for the nonempty skills in skill-ID order, minus used skill IDs and
    execution fingerprints. Aliases are numbered before the exclusion, so they keep gaps."""
    used_ids = {skill.skill_id for skill in used}
    used_fingerprints = {skill.fingerprint() for skill in used}
    nonempty = [skill for skill in skills if skill.nodes]
    return [(f"flow_{index:03d}", skill) for index, skill in enumerate(nonempty, start=1)
            if skill.skill_id not in used_ids and skill.fingerprint() not in used_fingerprints]


def later_round_prompt(state: dict, flows: list[tuple[str, ReflectionSkill]], precedents: str) -> str:
    listing = []
    for alias, skill in flows:
        listing += [f"### `{alias}`", "", f"- Communication flow: {communication_flow(skill)}",
                    "- After this communication, the Task Owner revises the current work once.", ""]
    return render("planner/later_round_retrieval.md", {
        "CURRENT_WORK_SUMMARY": state_text(state), "REVIEWER_PROFILES": reviewer_profiles(),
        "AVAILABLE_COMMUNICATION_FLOWS": "\n".join(listing).rstrip() + "\n", "PAST_STATE_EXAMPLES": precedents})


def check_later_round(reply: dict, allowed: list[str], count: int) -> None:
    ids = [row["option_id"] for row in reply["recommendations"]]
    if len(ids) != count + 1 or len(set(ids)) != len(ids) or ids.count(STOP) != 1 or not set(ids) - {STOP} <= set(allowed):
        raise ValueError(f"expected {count} unique available flows and {STOP} exactly once")


def select_later_round(profile: llm.ModelProfile, state: dict, library: Library, used: list[ReflectionSkill], *,
                       count: int = 9, temperature: float | None = None,
                       log_path: Path | None = None) -> tuple[str, dict]:
    """(skill ID of the highest-ranked flow, the ranking). leave_unchanged is ranked but never chosen."""
    flows = later_round_flows(library.skills(), used)
    if len(flows) < count:
        raise ValueError(f"later-round retrieval needs {count} unused nonempty skills; the library has {len(flows)}")
    aliases = dict(flows)
    reply = llm.complete_json(profile, system=SYSTEM["later_round"],
                              prompt=later_round_prompt(state, flows, library.precedents()),
                              schema=_schema("planner/later_round_retrieval.schema.json"), temperature=temperature,
                              log_path=log_path, validate=lambda value: check_later_round(value, list(aliases), count))
    chosen = next(row["option_id"] for row in reply["recommendations"] if row["option_id"] != STOP)
    return aliases[chosen].skill_id, reply


def _nonblank(value) -> bool:
    return isinstance(value, str) and bool(value.strip())
