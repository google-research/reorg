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
"""The judge's workspace (App. A.5.2): the request, the reference files, the
signed-point rubric, and the candidate and expert files under randomized
Response A/B labels, each with a file manifest.

The visible order of each pair is drawn from one seeded stream in cohort
order, so every judge sees the same labels and a rerun reproduces them.
"""

from __future__ import annotations

import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

from reorg.files import write_json


@dataclass(frozen=True)
class Plan:
    """One candidate-vs-expert comparison. mapping[label] = {"kind": "submission"
    or "expert", "source_dir": the candidate's submission dir or None}."""
    index: int
    attempt_id: str
    task_id: str
    mapping: dict


def plans(submissions: dict[str, Path], seed: int) -> list[Plan]:
    """submissions: task_id -> final/submission dir, in cohort order."""
    rng = random.Random(seed)
    out = []
    for index, (task_id, submission) in enumerate(submissions.items()):
        candidate = {"kind": "submission", "source_dir": str(submission)}
        expert = {"kind": "expert", "source_dir": None}
        first, second = (expert, candidate) if rng.random() < 0.5 else (candidate, expert)
        out.append(Plan(index, f"{index:06d}_{task_id}", task_id, {"A": first, "B": second}))
    return out


def _copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _stage_submission(source: Path, target: Path) -> list[dict]:
    files = []
    for path in sorted(p for p in source.rglob("*") if p.is_file()):
        relative = Path("files") / path.relative_to(source)
        _copy(path, target / relative)
        files.append({"path": relative.as_posix(), "description": relative.name})
    return files


def _stage_expert(task: dict, data: Path, target: Path) -> list[dict]:
    used: set[str] = set()
    files = []
    for path_text in task["deliverable_files"]:
        # Expert files are flattened into files/; a repeated basename gets a _2, _3, ... suffix.
        base = Path(path_text).name
        name, index = base, 2
        while name in used:
            name = f"{Path(base).stem}_{index}{Path(base).suffix}"
            index += 1
        used.add(name)
        _copy(data / path_text, target / "files" / name)
        files.append({"path": f"files/{name}", "description": name})
    return files


def _task_markdown(task: dict, references: list[str]) -> str:
    links = "\n".join(f"- `{path}`" for path in references) or "- None"
    return (
        f"# GDPval Task\n\n- Task ID: `{task['task_id']}`\n"
        f"- Sector: {task['sector']}\n- Occupation: {task['occupation']}\n\n"
        f"## Prompt\n\n{task['prompt']}\n\n## Local Reference Files\n\n{links}\n"
    )


def stage(plan: Plan, task: dict, data: Path, sandbox: Path, instruction: str) -> None:
    """Write the workspace for `plan`; task is the GDPval row, data the snapshot root."""
    sandbox.mkdir(parents=True)
    (sandbox / "scratch").mkdir()
    references = []
    for path_text in task["reference_files"]:
        parts = Path(path_text).parts
        target = Path("reference_files", *(parts[1:] if parts[0] == "reference_files" else parts))
        _copy(data / path_text, sandbox / target)
        references.append(target.as_posix())
    (sandbox / "task.md").write_text(_task_markdown(task, references), encoding="utf-8")
    (sandbox / "rubric_pretty.md").write_text(str(task["rubric_pretty"]) + "\n", encoding="utf-8")
    write_json(sandbox / "rubric.json", json.loads(task["rubric_json"]))
    counts = {}
    for label, side in plan.mapping.items():
        target = sandbox / f"response_{label}"
        target.mkdir()
        files = (_stage_expert(task, data, target) if side["kind"] == "expert"
                 else _stage_submission(Path(side["source_dir"]), target))
        write_json(target / "manifest.json", {"files": files})
        counts[label] = len(files)
    write_json(sandbox / "comparison.json", {
        "comparison_id": plan.attempt_id,
        "task_id": plan.task_id,
        "is_ab_swap": False,
        "allow_online_verification_requested": False,
        "network_tools_enabled": False,
        "responses": {label: {"manifest": f"response_{label}/manifest.json", "file_count": count}
                      for label, count in counts.items()},
    })
    (sandbox / "INSTRUCTION.md").write_text(instruction + "\n", encoding="utf-8")
