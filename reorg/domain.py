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
"""The Domain interface: everything ReOrg needs from a benchmark.

The method core never reads a benchmark directory. A domain adapter
(data/<name>/adapter.py) owns its data and turns a task ID into the task
packet x = (u, R) of Sec. 3.1.

    prepare        download the pinned source, verify it, build a task index
    adapt_ids      the ordered adaptation stream x_1..x_T           Alg. 1
    eval_ids       the evaluation cohort                            Sec. 5.1
    heldout_ids    eval_ids not used for adaptation                 App. A.5.1
    smoke_ids      two tasks for liveness checks
    load_task      task ID -> TaskPacket (task-facing inputs only)

Task-ID lists live in data/<name>/splits/; the benchmark data itself is
downloaded by `prepare` into downloads/<name>/ (git-ignored).
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from .files import read_ids

ROOT = Path(__file__).resolve().parent.parent
DOMAINS = ROOT / "data"                   # one folder per benchmark


@dataclass(frozen=True)
class TaskPacket:
    """x = (u, R): the request u and the reference files R.

    Rubrics and expert deliverables never enter a packet. Workers,
    stakeholders and the online ranking judge see only this (Sec. 5.1);
    the domain's evaluator reads the hidden material separately."""
    task_id: str
    request: str
    references: tuple[Path, ...] = ()
    deliverables: tuple[str, ...] = ()   # file names the task asks for, when it names them


class Domain:
    name: str = "base"
    root: Path                            # data/<name>
    data: Path                            # downloads/<name>

    # ---- data --------------------------------------------------------------
    def prepare(self) -> dict:
        """Download the pinned source into self.data and verify it."""
        raise NotImplementedError

    def load_task(self, task_id: str) -> TaskPacket:
        raise NotImplementedError

    # ---- task sets ---------------------------------------------------------
    def ids(self, name: str) -> list[str]:
        return read_ids(self.root / "splits" / f"{name}.txt")

    def adapt_ids(self) -> list[str]:
        return self.ids("adapt_20")

    def eval_ids(self) -> list[str]:
        raise NotImplementedError

    def heldout_ids(self) -> list[str]:
        adapt = set(self.adapt_ids())
        return [task_id for task_id in self.eval_ids() if task_id not in adapt]

    def smoke_ids(self) -> list[str]:
        return self.ids("smoke")

    TASK_SETS = ("adapt", "eval", "heldout", "smoke")

    def task_set(self, name: str) -> list[str]:
        if name not in self.TASK_SETS:
            raise ValueError(f"unknown task set {name!r}; choose from {self.TASK_SETS}")
        return getattr(self, f"{name}_ids")()


def load_module(path: Path) -> ModuleType:
    """Import data/<name>/<file>.py as data.<name>.<file>, so sibling modules
    of different benchmarks never collide."""
    name = f"data.{path.parent.name}.{path.stem}"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_domain(name: str, data_root: Path) -> Domain:
    """A fresh domain instance bound to <data_root>/<name>; instances never
    share cached task rows across data roots."""
    path = DOMAINS / name / "adapter.py"
    if not path.is_file():
        raise SystemExit(f"unknown domain {name!r} (no {path})")
    domain = type(load_module(path).DOMAIN)()
    domain.root = DOMAINS / name
    domain.data = Path(data_root).expanduser().resolve() / name
    return domain
