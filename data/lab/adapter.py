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
"""Harvey's Legal Agent Benchmark (LAB, Sec. 5.1): legal deliverables scored
criterion by criterion.

The paper evaluates the 200 tasks in test_200.txt (our seeded test split of
the pinned checkout) and adapts on the 20 tasks in adapt_20.txt, drawn from
the remaining train + valid pool. Each task names the files it expects;
only those files are submitted.
"""

from __future__ import annotations

from functools import cached_property
from pathlib import Path

from reorg.domain import Domain, TaskPacket, load_module

HERE = Path(__file__).resolve().parent
prepare_data = load_module(HERE / "prepare_data.py")


class LAB(Domain):
    name = "lab"

    def prepare(self) -> dict:
        return prepare_data.build_index(self.data)

    def eval_ids(self) -> list[str]:
        return self.ids("test_200")

    @property
    def harvey_root(self) -> Path:
        return self.data / "harvey-labs"

    @cached_property
    def _rows(self) -> dict[str, dict]:
        return {row["task_id"]: row for row in prepare_data.load_rows(self.data)}

    def load_task(self, task_id: str) -> TaskPacket:
        row = self._rows[task_id]
        return TaskPacket(
            task_id=task_id,
            request=str(row["prompt"]),
            references=tuple(self.harvey_root / path for path in row["reference_files"]),
            deliverables=tuple(row["deliverable_files"]),
        )


DOMAIN = LAB()
