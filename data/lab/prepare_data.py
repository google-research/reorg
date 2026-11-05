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
"""LAB data: a harvey-labs checkout at the pinned commit and its task index.

Layout under <data>/lab:

    harvey-labs/   the checkout (tasks/<practice area>/.../task.json + documents/)
    tasks.jsonl    one rubric-free row per task with its split; its sha256 is pinned

The test split is ours, not Harvey's: whole task families are assigned to
train / valid / test by stratified sampling (practice area, work type,
deliverable extensions) with a fixed seed. The paper evaluates on the 200 test
tasks and draws its 20 adaptation tasks from train + valid.
"""

from __future__ import annotations

import json
import math
import random
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from reorg.files import read_ids, read_json, read_jsonl, require_sha256, sha256_file, write_jsonl

HERE = Path(__file__).resolve().parent
PIN = read_json(HERE / "splits" / "manifest.json")


def checkout(dest: Path) -> Path:
    """Clone harvey-labs once and check out the pinned commit (rerunnable)."""
    root = dest / "harvey-labs"
    if not (root / ".git").is_dir():
        subprocess.run(["git", "clone", "--no-checkout", PIN["repository"], str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "checkout", "--force", "--detach", PIN["revision"]], check=True)
    return root


# ---- task discovery --------------------------------------------------------

SCENARIO_RE = re.compile(r"^(scenario|variant|matter)[-_]?\d+$", re.IGNORECASE)
OUTPUT_HEADER_RE = re.compile(r"(?im)^###\s*Output\s*:\s*$")
NUMBERED_FILENAME_RE = re.compile(r"(?:\(\d+\)|\d+[.)])\s*`([^`]+)`")


def _basename(value: str) -> str | None:
    name = re.sub(r"^\d+[.)]\s*", "", value.strip().lstrip("-* ")).strip("` ")
    if not re.fullmatch(r"[^/\\\s]+\.[A-Za-z0-9]{1,8}", name) or Path(name).name != name:
        return None
    return name


def _instruction_deliverables(instructions: str) -> list[str]:
    sections = OUTPUT_HEADER_RE.split(instructions)
    if len(sections) > 1:
        names: list[str] = []
        for line in sections[-1].splitlines():
            if not line.strip():
                continue
            name = _basename(line)
            if name:
                names.append(name)
            elif names:
                break
        if names:
            return names
    return [name for raw in NUMBERED_FILENAME_RE.findall(instructions) if (name := _basename(raw))]


def _deliverables(config: dict[str, Any]) -> tuple[list[str], str]:
    """The files the task asks for: from task.json, else parsed from the instructions."""
    value = config.get("deliverables")
    if isinstance(value, dict):
        return [str(name) for name in value.values()], "task_json"
    if isinstance(value, list):
        return [str(name) for name in value], "task_json"
    names = _instruction_deliverables(str(config["instructions"]))
    # One task asks for a single document without naming it; upstream scoring
    # reads every produced file, so it gets a fixed name.
    return (names, "instructions") if names else (["output.docx"], "fallback_output_docx")


def discover_tasks(harvey_root: Path) -> list[dict[str, Any]]:
    tasks_root = harvey_root / "tasks"
    rows: list[dict[str, Any]] = []
    for task_path in sorted(tasks_root.rglob("task.json")):
        config = json.loads(task_path.read_text(encoding="utf-8"))
        task_dir = task_path.parent
        task_id = task_dir.relative_to(tasks_root).as_posix()
        parts = task_id.split("/")
        deliverables, deliverable_source = _deliverables(config)
        document_paths = [path for path in sorted((task_dir / "documents").rglob("*")) if path.is_file()]
        work_type = str(config.get("work_type") or "unknown")
        suffixes = sorted(Counter(Path(name).suffix.lower() or "none" for name in deliverables).elements())
        rows.append({
            "task_id": task_id,
            "title": str(config["title"]),
            "prompt": str(config["instructions"]),
            "practice_area": parts[0],
            "work_type": work_type,
            "tags": [str(tag) for tag in config.get("tags") or []],
            "deliverable_files": deliverables,
            "deliverable_source": deliverable_source,
            "reference_files": [path.relative_to(harvey_root).as_posix() for path in document_paths],
            "reference_file_manifest": [
                {
                    "path": path.relative_to(harvey_root).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in document_paths
            ],
            "task_config_sha256": sha256_file(task_path),
            "criterion_count": len(config["criteria"]),
            # scenarios of one matter ("<matter>/scenario-2") form a family that is never split
            "family_id": "/".join(parts[:-1]) if SCENARIO_RE.match(parts[-1]) else task_id,
            "stratum": f"{parts[0]}::{work_type}::{'+'.join(suffixes)}",
        })
    return rows


# ---- family-aware stratified split -----------------------------------------

def _stratum_targets(rows: list[dict[str, Any]], target: int) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row["stratum"])] += 1
    raw = {name: target * count / len(rows) for name, count in counts.items()}
    allocated = {name: math.floor(value) for name, value in raw.items()}
    remaining = target - sum(allocated.values())
    order = sorted(counts, key=lambda name: (raw[name] - allocated[name], counts[name], name), reverse=True)
    for name in order[:remaining]:
        allocated[name] += 1
    return allocated


def _closest_group_subset(groups: list[list[dict[str, Any]]], target: int,
                          rng: random.Random) -> list[list[dict[str, Any]]]:
    """Subset-sum over shuffled families: the family set whose size is closest to `target`."""
    shuffled = list(groups)
    rng.shuffle(shuffled)
    cap = target + max((len(group) for group in shuffled), default=0)
    reachable: dict[int, tuple[int, ...]] = {0: ()}
    for index, group in enumerate(shuffled):
        for current, chosen in sorted(list(reachable.items()), reverse=True):
            candidate = current + len(group)
            if candidate <= cap and candidate not in reachable:
                reachable[candidate] = (*chosen, index)
    best = min(reachable, key=lambda value: (abs(value - target), value > target, value))
    return [shuffled[index] for index in reachable[best]]


def _choose_families(rows: list[dict[str, Any]], target: int, rng: random.Random) -> set[str]:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_family[str(row["family_id"])].append(row)
    by_stratum: dict[str, list[list[dict[str, Any]]]] = defaultdict(list)
    for family_rows in by_family.values():
        stratum = Counter(str(row["stratum"]) for row in family_rows).most_common(1)[0][0]
        by_stratum[stratum].append(family_rows)
    targets = _stratum_targets(rows, target)
    selected: set[str] = set()
    for stratum in sorted(by_stratum):
        for group in _closest_group_subset(by_stratum[stratum], targets.get(stratum, 0), rng):
            selected.add(str(group[0]["family_id"]))
    return selected


def assign_splits(rows: list[dict[str, Any]], *, test_size: int, valid_size: int,
                  seed: int) -> list[dict[str, Any]]:
    """Assign whole task families to test, then valid; the rest is train."""
    rng = random.Random(seed)
    test = _choose_families(rows, test_size, rng)
    valid = _choose_families([row for row in rows if str(row["family_id"]) not in test], valid_size, rng)
    return [{**row, "split": "test" if row["family_id"] in test else "valid" if row["family_id"] in valid
             else "train"} for row in rows]


# ---- index -----------------------------------------------------------------

def build_index(dest: Path) -> dict:
    """Write tasks.jsonl with the seeded split, check its pinned hash, and check
    that the shipped test_200.txt is that split's test set."""
    sizes = PIN["split_sizes"]
    rows = assign_splits(discover_tasks(checkout(dest)), test_size=sizes["test"],
                         valid_size=sizes["valid"], seed=PIN["split_seed"])
    write_jsonl(dest / "tasks.jsonl", rows)
    require_sha256(dest / "tasks.jsonl", PIN["tasks_sha256"])
    if set(read_ids(HERE / "splits" / "test_200.txt")) != {row["task_id"] for row in rows if row["split"] == "test"}:
        raise ValueError("test_200.txt differs from the seeded test split")
    return {split: sum(row["split"] == split for row in rows) for split in sizes}


def load_rows(dest: Path) -> list[dict]:
    return read_jsonl(dest / "tasks.jsonl")
