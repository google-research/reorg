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
"""GDPval (Sec. 5.1): professional deliverables judged against expert work.

The public set has 220 tasks; the paper's 185-task cohort is every task with
an expert deliverable. adapt_20.txt lists 20 of them in adaptation-stream
order and heldout_165.txt the other 165. Workers see the request, the
reference files and, as the expected deliverables, the dataset paths of the
expert deliverables (deliverable_files/<hash>/<name>).
The expert files themselves and the rubrics are for evaluation only.
"""

from __future__ import annotations

from functools import cached_property
from pathlib import Path

from reorg.domain import Domain, TaskPacket, load_module

HERE = Path(__file__).resolve().parent
prepare_data = load_module(HERE / "prepare_data.py")


class GDPval(Domain):
    name = "gdpval"

    def prepare(self) -> dict:
        prepare_data.download(self.data)
        return prepare_data.verify(self.data)

    def eval_ids(self) -> list[str]:
        return self.ids("cohort_185")

    def heldout_ids(self) -> list[str]:
        return self.ids("heldout_165")

    @cached_property
    def _rows(self) -> dict[str, dict]:
        return {row["task_id"]: row for row in prepare_data.load_rows(self.data)}

    def load_task(self, task_id: str) -> TaskPacket:
        row = self._rows[task_id]
        return TaskPacket(
            task_id=task_id,
            request=str(row["prompt"]),
            references=tuple(self.data / path for path in row["reference_files"]),
            deliverables=tuple(str(path) for path in row["deliverable_files"]),   # paths only, never the files
        )


DOMAIN = GDPval()
