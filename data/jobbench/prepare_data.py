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
"""JobBench data: the pinned JobBench/job-bench snapshot and its task index.

Layout under <data>/jobbench:

    raw/                   the snapshot (tasks.parquet, easy.parquet, dataset/, dataset_easy/)
    main -> raw/dataset    split views, so task paths read "main/..." and "easy/..."
    easy -> raw/dataset_easy
    tasks.jsonl            one row per task (pinned hash)

A task's visible reference files are the worker's inputs. Files under
files_required_to_search/ are hidden: only the guarded web search on main
tasks may return them. Rubrics are for evaluation only.
"""

from __future__ import annotations

import json
from pathlib import Path

from reorg.files import read_json, read_jsonl, require_sha256, write_jsonl

HERE = Path(__file__).resolve().parent
PIN = read_json(HERE / "splits" / "manifest.json")
SPLITS = ("main", "easy")


def download(dest: Path) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=PIN["repo_id"], repo_type=PIN["repo_type"], revision=PIN["revision"],
                      allow_patterns=PIN["allow_patterns"], local_dir=dest / "raw")


def _rubric_criteria(rubric: dict) -> list:
    raw = rubric.get("criterion") or []
    return [raw] if isinstance(raw, str) else list(raw)


def _split_rows(dest: Path, split: str) -> list[dict]:
    import pandas as pd

    raw_split = PIN["raw_split_dirs"][split]
    frame = pd.read_parquet(dest / "raw" / PIN["parquet_files"][split]["path"])
    rows = []
    for _, source in frame.sort_values(["occupation", "task_num"]).iterrows():
        occupation, task_num, task_id = str(source["occupation"]), int(source["task_num"]), str(source["task_id"])
        task_dir = Path(split) / occupation / f"task{task_num}"
        hidden_dir = dest / "raw" / raw_split / occupation / f"task{task_num}" / "files_required_to_search"
        rubrics = json.loads(str(source["rubric_json"])).get("rubrics") or []
        rows.append({
            "benchmark": "jobbench",
            "split": split,
            "task_id": task_id,
            "canonical_task_id": f"{split}/{task_id}",
            "occupation": occupation,
            "task_num": task_num,
            "prompt": str(source["prompt"]),
            "task_dir": task_dir.as_posix(),
            "prompt_path": (task_dir / "task_folder" / "TASK_INSTRUCTIONS.txt").as_posix(),
            "rubric_path": (task_dir / "RUBRICS.json").as_posix(),
            "task_card_path": (task_dir / "task_card.md").as_posix(),
            # parquet paths start with the raw split dir ("dataset/..."); rewrite to the split view
            "visible_reference_files": [Path(split).joinpath(*Path(str(ref)).parts[1:]).as_posix()
                                        for ref in source["reference_files"]],
            "reference_file_urls": [str(item) for item in source["reference_file_urls"]],
            "reference_file_hf_uris": [str(item) for item in source["reference_file_hf_uris"]],
            "hidden_search_files": [(task_dir / "files_required_to_search" / p.relative_to(hidden_dir)).as_posix()
                                    for p in sorted(hidden_dir.rglob("*")) if p.is_file()],
            "rubric_count": len(rubrics),
            "criteria_count": sum(len(_rubric_criteria(rubric)) for rubric in rubrics),
            "max_score": sum(float(rubric.get("weight") or 0) for rubric in rubrics),
        })
    return rows


def build_index(dest: Path) -> dict:
    """Write the split views and tasks.jsonl, and check the index against its pinned hash."""
    for split in SPLITS:
        link = dest / split
        link.unlink(missing_ok=True)
        link.symlink_to(Path("raw") / PIN["raw_split_dirs"][split], target_is_directory=True)
    rows = [row for split in SPLITS for row in _split_rows(dest, split)]
    write_jsonl(dest / "tasks.jsonl", rows)
    require_sha256(dest / "tasks.jsonl", PIN["prepared_tasks_sha256"])
    return {split: sum(row["split"] == split for row in rows) for split in SPLITS}


def load_rows(dest: Path) -> list[dict]:
    return read_jsonl(dest / "tasks.jsonl")
