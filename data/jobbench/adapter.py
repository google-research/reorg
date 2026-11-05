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
"""JobBench (Sec. 5.1): occupational tasks scored against task rubrics.

All 128 tasks (65 main + 63 easy) are evaluated. Task IDs carry their subset
("main/..." or "easy/...") because some raw IDs exist in both. adapt_20.txt
lists the in-domain adaptation stream (10 main + 10 easy) and
heldout_108.txt the remaining tasks.
"""

from __future__ import annotations

from functools import cached_property
from pathlib import Path

from reorg.domain import Domain, TaskPacket, load_module

HERE = Path(__file__).resolve().parent
prepare_data = load_module(HERE / "prepare_data.py")


class JobBench(Domain):
    name = "jobbench"

    def prepare(self) -> dict:
        prepare_data.download(self.data)
        return prepare_data.build_index(self.data)

    def eval_ids(self) -> list[str]:
        return self.ids("all_128")

    def heldout_ids(self) -> list[str]:
        return self.ids("heldout_108")

    @cached_property
    def _rows(self) -> dict[str, dict]:
        return {row["canonical_task_id"]: row for row in prepare_data.load_rows(self.data)}

    def load_task(self, task_id: str) -> TaskPacket:
        row = self._rows[task_id]
        return TaskPacket(
            task_id=task_id,
            request=str(row["prompt"]),
            references=tuple(self.data / path for path in row["visible_reference_files"]),
        )


DOMAIN = JobBench()
