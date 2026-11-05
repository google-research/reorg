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
"""Summarize of Alg. 1: one task's retrieval lesson for the planner knowledge M
(Sec. 4.2; App. A.2, "Library Learning and Artifact Ranking").

The task's comparisons become observations of a communication flow against a
comparison point within one ranking:

    promotion   every promotion candidate against the best-ranked retrieved skill
    search      every child against its parent, in each search epoch that ranked both

An observation is better, tied or worse by tier, and uncertain when either
artifact is hard-invalid. Up to 5 observations fill each of four outcome groups
(most and least effective structures from promotion; most and least effective
edits from search, falling back to uncertain edits when no edit was worse),
the most stable first: resolved, then more group appearances, then more
distinct opponents. The lesson-compaction prompt (P7) shows each with the
judges' decisive differences, and the model's summary becomes one lesson
section of lessons.md.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .. import llm, planner
from ..files import write_json
from ..reflection_skill.graph import RUN_ROLE_IDS, ReflectionSkill
from .mutate import Edit

SYSTEM = "Follow the self-contained steering insight-compaction document. Return only valid JSON."
SCHEMA = {
    "type": "object",
    "properties": {
        "task_summary": {
            "type": "object",
            "properties": {
                "domain": {"type": "string"},
                "high_level_task": {"type": "string"},
                "requested_output": {"type": "string"},
            },
            "required": ["domain", "high_level_task", "requested_output"],
        },
        "structure_helpfulness_analysis": {"type": "string"},
        "critical_feature_analysis": {"type": "string"},
    },
    "required": ["task_summary", "structure_helpfulness_analysis", "critical_feature_analysis"],
}
OBSERVATION_PREFIX = ("(The current row produces {label}, and the other artifacts are produced by other "
                      "evaluated communication flows)")
NO_OBSERVATION = "No eligible observation was established."
# Controller identifiers a lesson must not repeat (group-local artifact labels are allowed).
_INTERNAL = re.compile(r"(?:slot_\d+|candidate[_-]|group[_-]|[a-f0-9]{64})", re.IGNORECASE)


def flow(skill: ReflectionSkill) -> str:
    stages = planner.communication_flow(skill)
    return "the worker creates the initial artifact; " + (
        f"before revision 1, {stages}" if stages else "revision 1 proceeds without colleague communication")


def describe(edit: Edit) -> str:
    """An edit in model-facing wording, naming roles by their RUN_ROLE_IDS ("xfn partner v003")."""
    placement = "in parallel" if edit.placement == "parallel" else "sequentially"
    role = RUN_ROLE_IDS[edit.role].replace("_", " ") if edit.role else None
    if edit.family == "replace_role":
        return f"replace the colleague role before revision 1 with {role}"
    if edit.family == "relocate":
        return f"move one colleague contribution from before revision 1 to {placement} before revision 1"
    return f"add {role} {edit.action.replace('_', ' ')} {placement} before revision 1"


def relation(record: dict, subject: str, comparator: str) -> str:
    """subject against comparator in a completed ranking."""
    if {subject, comparator} & set(record["hard_invalid"]):
        return "uncertain"
    tier = {name: index for index, members in enumerate(record["tiers"]) for name in members}
    return "better" if tier[subject] < tier[comparator] else "worse" if tier[subject] > tier[comparator] else "tied"


def _observation(record: dict, stage: str, subject, comparator) -> dict:
    """subject and comparator are search.Candidate objects ranked in `record`."""
    found = relation(record, subject.candidate_id, comparator.candidate_id)
    statistics = record["statistics"][subject.candidate_id]
    return {"record_id": f"{record['epoch_id']}:{subject.candidate_id}:{comparator.candidate_id}", "stage": stage,
            "flow": flow(subject.skill), "comparator_flow": flow(comparator.skill),
            "edit": describe(subject.edit) if stage == "search" else None, "relation": found,
            "uncertainty": "resolved" if found != "uncertain" else "aggregation_uncertain",
            "appearances": statistics["appearances"], "opponents": statistics["distinct_opponents"],
            "differences": [[group["labels"][subject.candidate_id], group["decisive_difference"]]
                            for group in record["groups"]
                            if subject.candidate_id in group["candidates"] and comparator.candidate_id in group["candidates"]]}


def observations(outcome, promotion: dict) -> list[dict]:
    """Observations of a search.Outcome and its promotion ranking."""
    rows = []
    if promotion["status"] == "completed":
        best = next((min(name for name in tier if name in outcome.retrieved) for tier in promotion["tiers"]
                     if set(tier) & set(outcome.retrieved)), None)
        for tier in promotion["tiers"] if best else []:
            rows += [_observation(promotion, "promotion", outcome.candidates[name], outcome.candidates[best])
                     for name in tier if name != best]
    for record in outcome.rankings:
        if record["epoch_id"].startswith("search_") and record["status"] == "completed":
            ranked = set(record["candidates"]) - set(record["ineligible"])
            rows += [_observation(record, "search", child, outcome.candidates[child.parent])
                     for child in outcome.candidates.values()
                     if child.parent and child.candidate_id in ranked and child.parent in ranked]
    return rows


def select(rows: list[dict], per_group: int = 5) -> dict[str, list[dict]]:
    def ranked(stage: str, found: str) -> list[dict]:
        return sorted((row for row in rows if row["stage"] == stage and row["relation"] == found), reverse=True,
                      key=lambda row: (row["uncertainty"] == "resolved", row["appearances"], row["opponents"],
                                       row["record_id"]))

    return {"most_effective_structures": ranked("promotion", "better")[:per_group],
            "least_effective_structures": ranked("promotion", "worse")[:per_group],
            "most_effective_edits": ranked("search", "better")[:per_group],
            "least_effective_edits": (ranked("search", "worse") or ranked("search", "uncertain"))[:per_group]}


def _cell(text: str) -> str:
    return "<br>".join(text.replace("|", "\\|").splitlines())


def _differences(row: dict) -> str:
    if not row["differences"]:
        return "not recorded"
    return "<br><br>".join(f"{OBSERVATION_PREFIX.format(label='Artifact ' + label[-1])}<br><br>{_cell(text)}"
                           for label, text in row["differences"])


def _table(rows: list[dict], edits: bool) -> str:
    if not rows:
        return NO_OBSERVATION
    header = (["Communication edit", "Parent flow", "Child flow"] if edits
              else ["Communication flow", "Comparison flow"]) + ["Relation", "Coverage", "Uncertainty",
                                                                 "Observed differences"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in rows:
        cells = [row["edit"], row["comparator_flow"], row["flow"]] if edits else [row["flow"], row["comparator_flow"]]
        cells += [row["relation"], f"{row['appearances']} appearances; {row['opponents']} opponents",
                  row["uncertainty"], _differences(row)]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def prompt(task: str, sections: dict[str, list[dict]]) -> str:
    """P7 for one task; `task` is the planner's task query."""
    return planner.render("learning/lesson_compaction.md", {
        "TASK": task.strip(),
        "MOST_EFFECTIVE_STRUCTURES": _table(sections["most_effective_structures"], edits=False),
        "LEAST_EFFECTIVE_STRUCTURES": _table(sections["least_effective_structures"], edits=False),
        "MOST_EFFECTIVE_EDITS": _table(sections["most_effective_edits"], edits=True),
        "LEAST_EFFECTIVE_EDITS": _table(sections["least_effective_edits"], edits=True)})


def check(reply: dict) -> None:
    summary = reply["task_summary"]
    values = [*summary.values(), reply["structure_helpfulness_analysis"], reply["critical_feature_analysis"]]
    if (set(reply) != {"task_summary", "structure_helpfulness_analysis", "critical_feature_analysis"}
            or set(summary) != {"domain", "high_level_task", "requested_output"}
            or not all(str(value).strip() for value in values)):
        raise ValueError("the lesson must have exactly the requested nonempty fields")
    if _INTERNAL.search(json.dumps(reply, ensure_ascii=False)):
        raise ValueError("the lesson leaks an internal controller identifier")


def lesson(index: int, reply: dict) -> str:
    """The lessons.md section of the t-th adaptation task."""
    summary = reply["task_summary"]
    return "\n".join([f"## `task_{index:03d}` — {summary['high_level_task']}", "",
                      f"- Domain: {summary['domain']}",
                      f"- Requested output: {summary['requested_output']}",
                      f"- Structure helpfulness: {reply['structure_helpfulness_analysis']}",
                      f"- Critical communication features: {reply['critical_feature_analysis']}"])


def summarize(profile: llm.ModelProfile, task: str, outcome, promotion: dict, *, index: int, out_dir: Path,
              temperature: float | None = None, per_group: int = 5) -> str:
    """Compact the observations of task x_index into its lesson section; evidence, prompt and reply
    are kept under out_dir."""
    sections = select(observations(outcome, promotion), per_group)
    text = prompt(task, sections)
    write_json(out_dir / "evidence.json", sections)
    (out_dir / "prompt.md").write_text(text, encoding="utf-8", newline="\n")
    reply = llm.complete_json(profile, system=SYSTEM, prompt=text, schema=SCHEMA, temperature=temperature,
                                log_path=out_dir / "calls.jsonl", validate=check)
    write_json(out_dir / "reply.json", reply)
    return lesson(index, reply)
