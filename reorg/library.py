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
"""The skill library L and planner knowledge M on disk (Sec. 3.2, Alg. 1, App. A.2-A.3).

    <library>/skills/<skill_id>.json   a reflection skill s (ReflectionSkill.to_dict())
    <library>/records.jsonl            evidence events, one JSON object per line, in event order
    <library>/lessons.md               first-round retrieval lessons in M
    <library>/precedents.md            later-round precedents in M (written by later-round learning)

Promote (Alg. 1) adds a skill whose execution fingerprint is new and
reinforces the skill already holding that fingerprint otherwise, so L keeps
one skill per fingerprint. Every event is one records.jsonl row:

    skill_id     the skill's file stem under skills/
    fingerprint  ReflectionSkill.fingerprint(), the execution identity
    event        "seed" | "promoted" | "reinforced"
    task_id      the adaptation task x_t that promoted or rediscovered it (null for seeds)
    evidence     provenance from learning, e.g. the candidate and ranking behind the event

The first-round planner shows each skill's number of promoted and reinforced
events as its prior observations. Both knowledge files are model-facing text:
the planner inserts lessons.md (stripped) into the first-round prompt and
precedents.md as is into the later-round prompt. Inference only reads a
library; learning snapshots it at task boundaries.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Iterable

from .files import read_json, read_jsonl
from .reflection_skill.graph import ReflectionSkill

# The header of lessons.md, and its body until the first lesson.
LESSONS_HEADER = ("# Steering retrieval insights\n\n"
                  "This document contains observations from completed earlier tasks. "
                  "It is an input to retrieval only and is never shown to a task judge.")
NO_LESSONS = "No completed-task observations are available yet."


def task_alias(task_id: str) -> str:
    """A file-system-safe task name: skill IDs learned on JobBench and LAB (whose task IDs
    contain "/") and later-round group IDs start with it."""
    readable = re.sub(r"[^A-Za-z0-9_.-]+", "_", task_id.strip()).strip("._-")[:72].rstrip("._-") or "task"
    return f"{readable}__{hashlib.sha256(task_id.strip().encode('utf-8')).hexdigest()[:16]}"


class Library:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    @classmethod
    def create(cls, root: Path, seeds: Iterable[ReflectionSkill]) -> "Library":
        library = cls(root)
        (library.root / "skills").mkdir(parents=True)
        (library.root / "records.jsonl").touch()
        library._write("lessons.md", f"{LESSONS_HEADER}\n\n{NO_LESSONS}\n")
        for skill in seeds:
            library._store(skill)
            library._record(skill.skill_id, skill.fingerprint(), "seed", None, {})
        return library

    def snapshot(self, root: Path) -> "Library":
        shutil.copytree(self.root, root)
        return Library(root)

    # ---- L -----------------------------------------------------------------
    def skills(self) -> list[ReflectionSkill]:
        """All skills in skill-ID order (the order the planner numbers them in)."""
        loaded = (ReflectionSkill.from_dict(read_json(path)) for path in (self.root / "skills").glob("*.json"))
        return sorted(loaded, key=lambda skill: skill.skill_id)

    def skill(self, skill_id: str) -> ReflectionSkill:
        return ReflectionSkill.from_dict(read_json(self.root / "skills" / f"{skill_id}.json"))

    def records(self) -> list[dict]:
        return read_jsonl(self.root / "records.jsonl")

    def observations(self) -> Counter:
        """Promoted and reinforced events per skill ID."""
        return Counter(row["skill_id"] for row in self.records() if row["event"] != "seed")

    def find(self, fingerprint: str) -> str | None:
        return next((row["skill_id"] for row in self.records() if row["fingerprint"] == fingerprint), None)

    def add(self, skill: ReflectionSkill, task_id: str, evidence: dict) -> tuple[str, str]:
        """Promote one skill: returns ("promoted", its ID) or ("reinforced", the existing skill's ID)."""
        fingerprint = skill.fingerprint()
        existing = self.find(fingerprint)
        if existing is None:
            self._store(skill)
        event, skill_id = ("promoted", skill.skill_id) if existing is None else ("reinforced", existing)
        self._record(skill_id, fingerprint, event, task_id, evidence)
        return event, skill_id

    # ---- M -----------------------------------------------------------------
    def lessons(self) -> str:
        return (self.root / "lessons.md").read_text(encoding="utf-8")

    def add_lesson(self, markdown: str) -> None:
        """Append one task's lesson section, replacing the NO_LESSONS placeholder."""
        text = self.lessons().rstrip().removesuffix(NO_LESSONS).rstrip()
        self._write("lessons.md", f"{text}\n\n{markdown.strip()}\n")

    def precedents(self) -> str:
        return (self.root / "precedents.md").read_text(encoding="utf-8")

    def write_precedents(self, text: str) -> None:
        self._write("precedents.md", text)

    # ---- files -------------------------------------------------------------
    def _store(self, skill: ReflectionSkill) -> None:
        # "x": a skill ID names one skill for the life of the library.
        with (self.root / "skills" / f"{skill.skill_id}.json").open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(skill.to_dict(), ensure_ascii=False, indent=2) + "\n")

    def _record(self, skill_id: str, fingerprint: str, event: str, task_id: str | None, evidence: dict) -> None:
        row = {"skill_id": skill_id, "fingerprint": fingerprint, "event": event, "task_id": task_id, "evidence": evidence}
        with (self.root / "records.jsonl").open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _write(self, name: str, text: str) -> None:
        (self.root / name).write_text(text, encoding="utf-8", newline="\n")
