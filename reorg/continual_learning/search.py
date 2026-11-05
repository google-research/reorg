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
"""The inner loop of Alg. 1 for one adaptation task x_t, and Promote (App. A.2).

    run B (the retrieved skills) and rank their artifacts       the initial tournament
    for 3 search steps:
        C <- Mutate(B)                                          up to 3 children per beam skill
        execute C; rank the artifacts of B u C                  one groupwise ranking epoch
        B <- the 3 highest-ranked skills
    Promote: rerank 5 eligible, distinct candidates (the finalists, then the best initial
             candidates) and keep the top 3

A candidate execution is the full lifecycle: the worker's initial work
A_0 = W(x), one reflection round with the candidate skill, and the
submission that is judged. It is eligible only if every action succeeded (a
strict round). Executions run in parallel and are cached by execution
fingerprint within the task. A ranking epoch needs 5 eligible candidates;
otherwise the beam carries over. Hard-invalid artifacts never enter the beam
or the library. When a tier straddles the remaining capacity, fingerprints not
yet selected are preferred and remaining ties go to the smallest fingerprint.
"""

from __future__ import annotations

import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Callable

from ..files import read_json, write_json
from ..reflection_skill.chronicle import WorkChronicle
from ..reflection_skill.graph import ReflectionSkill
from ..reflection_skill.reflect import Action, Runtime, reflect, work
from . import mutate, ranking
from .mutate import Edit

MIN_POOL = 5                                        # candidates a ranking epoch needs


@dataclass
class Candidate:
    candidate_id: str
    skill: ReflectionSkill
    parent: str | None = None
    edit: Edit | None = None
    artifact: Path | None = None                    # the judged submission; None unless the execution was strict

    @cached_property
    def fingerprint(self) -> str:
        return self.skill.fingerprint()


@dataclass
class Outcome:
    status: str                                     # completed | skipped (no beam of two after the tournament)
    candidates: dict[str, Candidate]                # retrieved, then children by round and parent
    retrieved: list[str]                            # the retrieved candidates B
    initial_order: list[str]                        # eligible retrieved candidates, best first by the tournament
    finalists: list[str]
    rankings: list[dict] = field(default_factory=list)


# ---- execution ---------------------------------------------------------------

def execute_skill(skill: ReflectionSkill, runtime: Runtime, submit: Callable[[Path, Path], list[str]], out_dir: Path,
                  parallelism: int, retries: int) -> Path | None:
    """A_0, one round with `skill`, and the submission (out_dir/submission), or None if any action
    failed. The outcome is kept in out_dir/result.json and reused."""
    if (out_dir / "result.json").exists():
        return out_dir / "submission" if read_json(out_dir / "result.json")["eligible"] else None
    shutil.rmtree(out_dir, ignore_errors=True)      # an interrupted execution starts over
    chronicle = WorkChronicle(out_dir / "work_chronicle.md")
    initial = work(Action("worker_initial", "initial_work", None), runtime, None, chronicle, out_dir / "a0", retries)
    status, eligible = f"initial_{initial.status}", False
    if initial.status == "ok":
        result = reflect(skill, runtime, initial.artifact, chronicle, out_dir / "round_1", parallelism, retries)
        write_json(out_dir / "round_1" / "round.json", result.to_dict())
        status = result.status
        eligible = status == "strict" and bool(submit(result.artifact, out_dir / "submission"))
    write_json(out_dir / "result.json", {"skill": skill.to_dict(), "status": status, "eligible": eligible})
    return out_dir / "submission" if eligible else None


def executor(runtime: Runtime, submit: Callable[[Path, Path], list[str]], out_dir: Path, *, parallelism: int = 9,
             stakeholder_parallelism: int = 6, retries: int = 2) -> Callable[[list[Candidate]], None]:
    """execute(candidates): run each distinct fingerprint once, concurrently, under
    out_dir/<fingerprint[:16]>/, and set every candidate's artifact."""
    def execute(candidates: list[Candidate]) -> None:
        unique = {candidate.fingerprint: candidate.skill for candidate in candidates}
        with ThreadPoolExecutor(max_workers=max(1, min(parallelism, len(unique)))) as pool:
            artifacts = dict(zip(unique, pool.map(
                lambda item: execute_skill(item[1], runtime, submit, out_dir / item[0][:16], stakeholder_parallelism,
                                           retries), unique.items())))
        for candidate in candidates:
            candidate.artifact = artifacts[candidate.fingerprint]
    return execute


# ---- ranking -----------------------------------------------------------------

def ranker(task: ranking.Material, judge: ranking.Judge, out_dir: Path,
           seed: int = ranking.SEED) -> Callable[..., dict]:
    """rank(epoch_id, pool, families=None): the ranking record of a pool, written to
    out_dir/<epoch_id>.json. families maps a beam parent to its children, which must share a group
    with it. With fewer than MIN_POOL eligible candidates the record is "skipped"."""
    def rank(epoch_id: str, pool: list[Candidate], families: dict[str, list[str]] | None = None) -> dict:
        eligible = {candidate.candidate_id: candidate.artifact for candidate in pool if candidate.artifact}
        record = {"epoch_id": epoch_id, "status": "skipped", "candidates": [c.candidate_id for c in pool],
                  "ineligible": [c.candidate_id for c in pool if not c.artifact]}
        if len(eligible) >= MIN_POOL:
            families = {parent: [child for child in children if child in eligible]
                        for parent, children in (families or {}).items() if parent in eligible}
            groups = ranking.schedule(epoch_id, list(eligible), seed,
                                      [(parent, child) for parent, children in families.items() for child in children],
                                      families)
            record.update(status="completed", **ranking.rank(groups, eligible, task=task, judge=judge,
                                                             out_dir=out_dir / epoch_id))
        write_json(out_dir / f"{epoch_id}.json", record)
        return record
    return rank


def select(record: dict, candidates: dict[str, Candidate], capacity: int) -> list[Candidate]:
    """The top `capacity` candidates of a completed ranking, without hard-invalid artifacts."""
    selected: list[Candidate] = []
    for tier in record["tiers"]:
        available = sorted((candidates[name] for name in tier if name not in record["hard_invalid"]),
                           key=lambda candidate: candidate.fingerprint)
        if len(selected) + len(available) <= capacity:
            selected += available
            if len(selected) == capacity:
                break
            continue
        while len(selected) < capacity:              # the tier straddles the capacity
            taken = {candidate.fingerprint for candidate in selected}
            chosen = next((c for c in available if selected and c.fingerprint not in taken), available[0])
            selected.append(chosen)
            available.remove(chosen)
        break
    return selected


# ---- Alg. 1 ------------------------------------------------------------------

def run(retrieved: list[ReflectionSkill], *, execute: Callable[[list[Candidate]], None], rank: Callable[..., dict],
        beam_width: int = 3, children_per_parent: int = 3, iterations: int = 3, seed: int = 0,
        weights: dict[str, float] = mutate.WEIGHTS) -> Outcome:
    candidates = {f"retrieved_{k:03d}": Candidate(f"retrieved_{k:03d}", skill) for k, skill in enumerate(retrieved, 1)}
    initial = list(candidates.values())
    execute(initial)
    eligible = [candidate for candidate in initial if candidate.artifact]
    rankings = [rank("initial", initial)]
    if rankings[0]["status"] == "completed":
        beam = select(rankings[0], candidates, beam_width)
        order = select(rankings[0], candidates, len(eligible))
    else:                                           # too few artifacts to rank: keep the retrieval order
        beam, order = eligible[:beam_width], eligible
    outcome = Outcome("skipped", candidates, list(candidates), [c.candidate_id for c in order], [], rankings)
    if len(beam) <= 1:
        outcome.finalists = [c.candidate_id for c in beam]
        return outcome

    seen = {candidate.fingerprint for candidate in initial}
    for round_index in range(1, iterations + 1):
        children, families = [], {}
        for parent in beam:
            found = [Candidate(candidate_id, skill, parent.candidate_id, edit) for candidate_id, skill, edit in
                     mutate.children(parent.skill, parent.candidate_id, round_index, limit=children_per_parent,
                                     seen=seen, seed=seed, weights=weights)]
            children += found
            families[parent.candidate_id] = [child.candidate_id for child in found]
            candidates.update((child.candidate_id, child) for child in found)
        execute(children)
        record = rank(f"search_{round_index:03d}", beam + children, families)
        rankings.append(record)
        if record["status"] == "completed":
            selected = select(record, candidates, beam_width)
            if not selected:
                break
            beam = selected
    outcome.status, outcome.finalists = "completed", [c.candidate_id for c in beam]
    return outcome


def promote(outcome: Outcome, *, rank: Callable[..., dict], pool_size: int = 5,
            top: int = 3) -> tuple[dict, list[Candidate]]:
    """Rerank the finalists and the best initial candidates (eligible, one per fingerprint, at most
    pool_size) and return the ranking and its top `top` candidates."""
    pool, seen = [], set()
    for candidate_id in [*outcome.finalists[:top], *outcome.initial_order]:
        candidate = outcome.candidates[candidate_id]
        if candidate.artifact and candidate.fingerprint not in seen and len(pool) < pool_size:
            seen.add(candidate.fingerprint)
            pool.append(candidate)
    record = rank("promotion", pool)
    return record, select(record, outcome.candidates, top) if record["status"] == "completed" else []
