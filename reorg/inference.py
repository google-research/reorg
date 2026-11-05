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
"""Inference: the reflection loop of Eq. 1 for one task (Sec. 3.2; App. A.3, "Deployment").

    A_0 = W(x)
    for r = 1, ..., R (R = 3):
        r > 1:  project (x, H_{r-1}) to a work state; stop if the completion gate scores >= 9
        s_r  =  first-round retrieval (r = 1) or later-round retrieval (r > 1) from the frozen (L, M)
        F_r  =  Reflect(x, A_{r-1}, H_{r-1}; s_r);   A_r = W(x, A_{r-1}, H_{r-1}, F_r)

The loop also ends after a round that is not strict (a stakeholder or the
worker failed): reflection continues only from strict states. The library is
read-only here. A domain plugs in through its runtime module
(data/<name>/runtime.py), passed as `hooks`:

    make_runtime(domain, task_id, profile, config) -> Runtime
    submission(domain, task_id, artifact_dir, out_dir) -> list[str]

Run layout, read by the judges:

    runs/<benchmark>/<run_id>/run.json
    runs/<benchmark>/<run_id>/<task_id>/a0/                initial work
                                     round_<r>/         planner.json, planner_calls.jsonl,
                                                        nodes/, artifact/, round.json
                                     final/submission/  the judged files
                                     work_chronicle.md
                                     result.json
"""

from __future__ import annotations

import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from . import planner
from .domain import Domain, TaskPacket
from .files import write_json
from .library import Library
from .llm import ModelProfile
from .reflection_skill.chronicle import WorkChronicle
from .reflection_skill.graph import ReflectionSkill
from .reflection_skill.reflect import Action, reflect, work


def plan(r: int, packet: TaskPacket, data_root: Path, chronicle: WorkChronicle, library: Library,
         used: list[ReflectionSkill], profile: ModelProfile, config: dict, log_path: Path) -> dict:
    """The planner's decision before round r: the ranking and the chosen "skill_id", plus the
    work "state" and "gate" result from round 2 on. No "skill_id" means the gate stopped."""
    calls = {"temperature": config["planner_temperature"].get(profile.provider), "log_path": log_path}
    if r == 1:
        ranking = planner.select_first_round(profile, planner.task_query(packet, data_root), library,
                                             top_k=config["first_round_top_k"], **calls)
        return {"round": r, "ranking": ranking, "skill_id": ranking[0][0]}
    state = planner.project_work_state(profile, packet.request, chronicle.text(), **calls)
    gate = planner.completion_gate(profile, state, **calls)
    if gate["task_completion_score"] >= config["stop_score"]:
        return {"round": r, "state": state, "gate": gate}
    skill_id, ranking = planner.select_later_round(profile, state, library, used, count=config["later_round_flows"],
                                                   **calls)
    return {"round": r, "state": state, "gate": gate, "ranking": ranking, "skill_id": skill_id}


def run_task(domain: Domain, task_id: str, *, hooks: Any, library: Library, worker: ModelProfile,
             planner_profile: ModelProfile, config: dict, task_dir: Path) -> dict:
    packet = domain.load_task(task_id)
    shutil.rmtree(task_dir, ignore_errors=True)     # a rerun under the same run ID starts the task afresh
    runtime = hooks.make_runtime(domain, task_id, worker, config)
    chronicle = WorkChronicle(task_dir / "work_chronicle.md")
    initial = work(Action("worker_initial", "initial_work", None), runtime, None, chronicle, task_dir / "a0",
                   config["retries"])
    artifact = initial.artifact
    rounds, used = [], []
    stop, budget = ("max_rounds", config["max_rounds"]) if artifact else ("initial_work_failed", 0)

    for r in range(1, budget + 1):
        round_dir = task_dir / f"round_{r}"
        decision = plan(r, packet, domain.data, chronicle, library, used, planner_profile, config,
                        round_dir / "planner_calls.jsonl")
        write_json(round_dir / "planner.json", decision)
        row = {"round": r, "completion_score": decision["gate"]["task_completion_score"] if r > 1 else None}
        if "skill_id" not in decision:
            rounds.append(row)
            stop = "completion_gate"
            break
        skill = library.skill(decision["skill_id"])
        result = reflect(skill, runtime, artifact, chronicle, round_dir, config["stakeholder_parallelism"],
                         config["retries"])
        write_json(round_dir / "round.json", result.to_dict())
        used.append(skill)
        artifact = result.artifact
        rounds.append({**row, "skill_id": skill.skill_id, "status": result.status})
        if result.status != "strict":
            stop = f"round_{result.status}"
            break

    files = hooks.submission(domain, task_id, artifact, task_dir / "final" / "submission") if artifact else []
    # The judges' vocabulary: "complete" is judged, "infra_error" is unavailable, anything else scores zero.
    status = "complete" if files else "infra_error" if initial.status == "infrastructure_error" else "incomplete"
    summary = {"task_id": task_id, "status": status, "stop": stop, "rounds": rounds,
               "artifact": str(artifact) if artifact else None, "submission": files}
    write_json(task_dir / "result.json", summary)
    return summary


def run_tasks(domain: Domain, task_ids: list[str], *, run_dir: Path, hooks: Any, library: Library,
              worker: ModelProfile, planner_profile: ModelProfile, config: dict) -> list[dict]:
    """Run tasks concurrently, each under run_dir/<task_id>/, and write run.json."""
    def one(task_id: str) -> dict:
        try:
            return run_task(domain, task_id, hooks=hooks, library=library, worker=worker,
                            planner_profile=planner_profile, config=config, task_dir=run_dir / task_id)
        except Exception as exc:                    # one failed task must not end the run
            return {"task_id": task_id, "status": "error", "error": repr(exc)}

    with ThreadPoolExecutor(max_workers=config["task_parallelism"]) as pool:
        results = list(pool.map(one, task_ids))
    write_json(run_dir / "run.json", {"domain": domain.name, "run_id": run_dir.name, "library": str(library.root),
                                      "worker": worker.name, "planner": planner_profile.name, "config": config,
                                      "tasks": results})
    return results
