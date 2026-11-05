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
"""GDPval evaluation (App. A.5.2): a three-judge pairwise panel against the
expert deliverable.

Gemini 3.1 Pro, Qwen 3.6 27B and Claude Sonnet 4.6 each inspect a fresh
workspace holding the request, references, signed-point rubric, and the
candidate and expert files under randomized Response A/B labels,
using code execution and image viewing without web access (runner.py). Their
categorical majority decides win / tie / loss; a split is a tie (vote.py):

    WinTie = (N_win + N_tie) / N_cohort.

Settings live in configs/judges/gdpval_*.json, and the members need
GEMINI_API_KEY, VLLM_BASE_URL and VERTEX_PROJECT (providers.py).

    evaluate(run_dir, task_ids, data) reads  <run_dir>/<task_id>/final/submission/
                                      writes <run_dir>/eval/<member>/results.jsonl, majority.jsonl, summary.json
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from reorg.domain import DOMAINS, ROOT, load_module
from reorg.files import read_json, write_json, write_jsonl

from . import runner, sandbox, vote

CONFIGS = ROOT / "configs" / "judges"


def evaluate(run_dir: Path, task_ids: list[str], data: Path) -> dict:
    """Judge every task's submission against its expert deliverable; returns
    win / tie / loss / unavailable counts and WinTie over len(task_ids)."""
    missing = [name for name in ("GEMINI_API_KEY", "VLLM_BASE_URL", "VERTEX_PROJECT") if name not in os.environ]
    if missing:
        raise SystemExit(f"the GDPval judge panel needs {', '.join(missing)}")
    panel = read_json(CONFIGS / "gdpval_trio.json")
    members = {name: read_json(CONFIGS / f"gdpval_{name}.json") for name in panel["members"]}
    out = run_dir / "eval"
    rows = {row["task_id"]: row for row in load_module(DOMAINS / "gdpval" / "prepare_data.py").load_rows(data)}
    # A task with no submitted file is not judged; it stays in the denominator as unavailable.
    submissions = {task_id: run_dir / task_id / "final" / "submission" for task_id in task_ids}
    submissions = {task_id: path for task_id, path in submissions.items()
                   if path.is_dir() and any(item.is_file() for item in path.rglob("*"))}
    plans = sandbox.plans(submissions, panel["seed"])

    results: dict[str, list[dict]] = {name: [] for name in members}
    with ProcessPoolExecutor(panel["parallelism"]) as pool:
        futures = [pool.submit(runner.judge, name, profile, panel, plan, rows[plan.task_id], data, out)
                   for name, profile in members.items() for plan in plans]
        for future in as_completed(futures):
            row = future.result()
            results[row["member"]].append(row)
    for name, judged in results.items():
        write_jsonl(out / name / "results.jsonl", sorted(judged, key=lambda row: row["attempt_id"]))

    decisions = vote.majority(results)
    write_jsonl(out / "majority.jsonl", [{"task_id": task_id, **decision} for task_id, decision in decisions.items()])
    summary = vote.win_tie(decisions, task_ids)
    write_json(out / "summary.json", summary)
    return summary
