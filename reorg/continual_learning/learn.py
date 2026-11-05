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
"""Alg. 1: continual reflection-skill learning over the adaptation stream x_1..x_T (Sec. 4).

    for t = 1, ..., T:
        B   <- Retrieve(x_t, L_{t-1}, M_{t-1})          first-round planner (P5), top 5
        search B; L_t <- Promote(L_{t-1}, B)            search.run, search.promote: each of the
                                                        top 3 is added, or reinforces the skill
                                                        holding its execution fingerprint
        M_t <- M_{t-1} u Summarize(x_t, outcomes)       knowledge.summarize (P7)

Tasks run in stream order, each from the previous boundary's (L, M). One model
serves as worker, stakeholders, planner, judge and note taker. Layout:

    <out>/snapshots/00_seeds/            (L_0, M_0)
    <out>/snapshots/<t>_<task_id>/       (L_t, M_t)
    <out>/library/                       the latest snapshot, for inference
    <out>/tasks/<task_id>/               retrieval.jsonl, executions/, rankings/<epoch>.json,
                                         candidates.json, compaction/, record.json

record.json marks a finished task. A rerun resumes at the first unfinished
task, rebuilding its snapshot from the previous boundary and reusing the
executions and group judgments already on disk.
"""

from __future__ import annotations

import shutil
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from .. import planner
from ..domain import Domain
from ..files import read_json, write_json
from ..library import Library, task_alias
from ..llm import ModelProfile
from . import knowledge, ranking, search
from .seeds import SEEDS


def decide(library: Library, task_id: str, promotion: dict, selected: list[search.Candidate], top: int = 3) -> dict:
    """Promote: add each of the top `top` promotion candidates to L, or reinforce the skill holding its
    fingerprint. No update unless the promotion ranking yields exactly `top` selectable candidates."""
    if len(selected) != top:
        return {"status": "no_update", "selected": [c.candidate_id for c in selected], "promoted": [],
                "reinforced": [], "reason": f"promotion ranking {promotion['status']} with {len(selected)} selectable"}
    promoted, reinforced = [], []
    # Skill IDs start with the raw task ID, or with its alias when the ID contains "/" (JobBench, LAB).
    prefix = task_alias(task_id) if "/" in task_id else task_id
    for candidate in selected:
        event, skill_id = library.add(replace(candidate.skill, skill_id=f"{prefix}_{candidate.candidate_id}"),
                                      task_id, {"candidate_id": candidate.candidate_id, "epoch_id": "promotion"})
        (promoted if event == "promoted" else reinforced).append(skill_id)
    return {"status": "updated", "selected": [c.candidate_id for c in selected], "promoted": promoted,
            "reinforced": reinforced, "reason": None}


def learn_task(domain: Domain, task_id: str, index: int, *, library: Library, task_dir: Path, hooks: Any,
               profile: ModelProfile, config: dict) -> dict:
    """Alg. 1 for x_t (t = index), updating `library` from L_{t-1} to L_t and M_{t-1} to M_t."""
    packet = domain.load_task(task_id)
    query = planner.task_query(packet, domain.data)
    temperature = config["temperature"].get(profile.provider)
    # A resumed task keeps its first retrieval: cached executions and judgments refer to its candidate IDs.
    if (task_dir / "retrieved.json").exists():
        retrieved = [tuple(row) for row in read_json(task_dir / "retrieved.json")]
    else:
        retrieved = planner.select_first_round(profile, query, library, top_k=config["retrieve_top_k"],
                                               temperature=temperature, log_path=task_dir / "retrieval.jsonl")
        write_json(task_dir / "retrieved.json", retrieved)

    runtime = hooks.make_runtime(domain, task_id, profile, config)
    execute = search.executor(runtime, lambda artifact, out: hooks.submission(domain, task_id, artifact, out),
                              task_dir / "executions", parallelism=config["candidate_parallelism"],
                              stakeholder_parallelism=config["stakeholder_parallelism"], retries=config["retries"])
    judge = ranking.Judge(replace(profile, temperature=profile.temperature if temperature is None else temperature),
                          **config["judge"])
    rank = search.ranker(ranking.material(packet, domain.data), judge, task_dir / "rankings", config["grouping_seed"])
    outcome = search.run([library.skill(skill_id) for skill_id, _ in retrieved], execute=execute, rank=rank,
                         beam_width=config["beam_width"], children_per_parent=config["children_per_parent"],
                         iterations=config["iterations"], seed=config["mutation_seed"],
                         weights=config["mutation_weights"])
    promotion, selected = search.promote(outcome, rank=rank, pool_size=config["promotion_pool"],
                                         top=config["promote_top"])
    write_json(task_dir / "candidates.json", {name: {
        "skill": c.skill.to_dict(), "fingerprint": c.fingerprint, "parent": c.parent,
        "edit": asdict(c.edit) if c.edit else None, "eligible": c.artifact is not None}
        for name, c in outcome.candidates.items()})

    decision = decide(library, task_id, promotion, selected, config["promote_top"])
    lesson = knowledge.summarize(profile, query, outcome, promotion, index=index, out_dir=task_dir / "compaction",
                                 temperature=temperature, per_group=config["lessons_per_group"])
    library.add_lesson(lesson)
    return {"task_id": task_id, "index": index, "retrieved": retrieved, "search": outcome.status,
            "initial_order": outcome.initial_order, "finalists": outcome.finalists,
            "rankings": [record["epoch_id"] for record in outcome.rankings] + ["promotion"],
            "promotion": promotion["status"], "decision": decision, "lesson": lesson}


def run(domain: Domain, task_ids: list[str], *, out_dir: Path, hooks: Any, profile: ModelProfile,
        config: dict) -> list[dict]:
    """Learn (L_T, M_T) from the ordered stream; returns the per-task records."""
    previous = out_dir / "snapshots" / "00_seeds"
    if not previous.exists():
        Library.create(previous, SEEDS)
    records = []
    for index, task_id in enumerate(task_ids, start=1):
        boundary, task_dir = out_dir / "snapshots" / f"{index:02d}_{task_id}", out_dir / "tasks" / task_id
        if not (task_dir / "record.json").exists():
            shutil.rmtree(boundary, ignore_errors=True)
            library = Library(previous).snapshot(boundary)
            record = learn_task(domain, task_id, index, library=library, task_dir=task_dir, hooks=hooks,
                                profile=profile, config=config)
            shutil.rmtree(out_dir / "library", ignore_errors=True)
            shutil.copytree(boundary, out_dir / "library")
            write_json(task_dir / "record.json", record)
        records.append(read_json(task_dir / "record.json"))
        previous = boundary
    return records
