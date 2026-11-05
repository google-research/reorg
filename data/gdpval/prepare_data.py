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
"""GDPval data: the pinned openai/gdpval snapshot.

download() fetches the snapshot; verify() checks the parquet hash, one digest
over every reference and expert deliverable file, and that the 185-task
cohort is exactly the tasks with expert deliverables, in parquet order.
"""

from __future__ import annotations

from pathlib import Path

from reorg.files import files_digest, read_ids, read_json, require_sha256, sha256_file

HERE = Path(__file__).resolve().parent
PIN = read_json(HERE / "splits" / "manifest.json")


def download(dest: Path) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=PIN["repo_id"], repo_type="dataset", revision=PIN["revision"], local_dir=dest)


def load_rows(dest: Path) -> list[dict]:
    import pandas as pd

    return pd.read_parquet(dest / PIN["parquet"]).to_dict(orient="records")


def cohort(rows: list[dict]) -> list[str]:
    """The judgeable cohort: tasks that come with an expert deliverable."""
    return [row["task_id"] for row in rows if len(row["deliverable_files"])]


def verify(dest: Path) -> dict:
    require_sha256(dest / PIN["parquet"], PIN["parquet_sha256"])
    rows = load_rows(dest)
    paths = {PIN["parquet"]}
    for row in rows:
        paths.update(row["reference_files"])
        paths.update(row["deliverable_files"])
    if files_digest({path: sha256_file(dest / path) for path in paths}) != PIN["files_digest"]:
        raise ValueError("reference or expert files differ from the pinned digest")
    if cohort(rows) != read_ids(HERE / "splits" / "cohort_185.txt"):
        raise ValueError("cohort_185.txt differs from the tasks with expert deliverables")
    return {"tasks": len(rows), "files": len(paths), "cohort": len(cohort(rows))}
