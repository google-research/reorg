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
"""The work chronicle H_r (Sec. 3.1, App. A.1).

One shared Markdown file records progress and prior advice across actions and
rounds. Each action sees a copy at collaboration_workspace/work_chronicle.md
and may append one "## Work Chronicle Update" section. The controller merges
that section into the canonical chronicle under a label naming the actor and
action. A missing or malformed update never blocks the round: the controller
substitutes a fallback (the stakeholder's handoff summary, or the status).
"""

from __future__ import annotations

import re
import threading
from pathlib import Path

INITIAL = ("# Work Chronicle\n\nThis is the shared working memory for the current deliverable.\n\n"
           "## Prior Updates\n\nNo action updates yet.\n")
STAGED = "collaboration_workspace/work_chronicle.md"

# Action labels shown to the models in the chronicle.
ACTION_LABELS = {"initial_work": "First Iter Work", "revise": "Improve Work",
                 "feedback": "Feedback", "propose": "Propose", "loop_in": "Loop In"}

# "## Work Chronicle Update", tolerating heading-level and suffix variants ("### ...", "... - round 2").
_HEADING = re.compile(r"#{2,3}[ \t]+Work Chronicle Update([ \t]*[-:()\[\]–—].*)?", re.IGNORECASE)


def extract_update(appended: str) -> str:
    """The body of the one update section an action appended, or "" if there is none.

    Models sometimes mirror the merged format ("## <label>" then "### Work
    Chronicle Update"); that single wrapper line is accepted and dropped."""
    lines = appended.splitlines()
    headings = [i for i, line in enumerate(lines) if _HEADING.fullmatch(line.strip())]
    if len(headings) != 1:
        return ""
    prefix = [line.strip() for line in lines[:headings[0]] if line.strip()]
    if prefix and not (len(prefix) == 1 and prefix[0].startswith("## ") and lines[headings[0]].lstrip().startswith("###")):
        return ""
    return "\n".join(lines[headings[0] + 1:]).strip()


def handoff_summary(text: str) -> str:
    """The body of a contribution's "# Handoff Summary" section."""
    match = re.search(r"^# Handoff Summary[ \t]*\n(.*?)(?=^# |\Z)", text, re.MULTILINE | re.DOTALL)
    return match.group(1).strip() if match else ""


def label(display_name: str, action: str, vertex_id: str) -> str:
    return f"{display_name} - {ACTION_LABELS[action]} - {vertex_id}"


class WorkChronicle:
    def __init__(self, path: Path, text: str = INITIAL) -> None:
        self.path = path
        self._lock = threading.Lock()                # stakeholders of one layer merge concurrently
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")

    def text(self) -> str:
        return self.path.read_text(encoding="utf-8")

    def stage(self, sandbox: Path) -> str:
        """Copy the chronicle into an action sandbox; returns the staged text."""
        base = self.text()
        (sandbox / STAGED).parent.mkdir(parents=True, exist_ok=True)
        (sandbox / STAGED).write_text(base, encoding="utf-8", newline="\n")
        return base

    def merge(self, sandbox: Path, base: str, label: str, fallback: str) -> None:
        """Append the action's update (whatever it added after `base`), else the fallback."""
        staged = sandbox / STAGED
        post = staged.read_text(encoding="utf-8", errors="replace") if staged.is_file() else ""
        update = extract_update(post[len(base):] if post.startswith(base) else post) or fallback.strip()
        if update:
            self.append(label, update)

    def append(self, label: str, update: str) -> None:
        with self._lock:
            existing = self.text().rstrip().replace("\nNo action updates yet.", "")
            entry = f"\n## {label}\n\n### Work Chronicle Update\n\n{update.strip()}\n"
            self.path.write_text(existing + "\n" + entry, encoding="utf-8", newline="\n")
