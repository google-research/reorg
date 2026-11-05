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
"""Prompt templates (App. A.6) and stakeholder profiles (App. A.7).

manifest.json (in this folder) records each file's sha256 and paper box (P1-P11,
R1-R10), and tests/test_core.py checks the files against it. In model-facing
text, "task owner" is the worker W and "communication structure" is a
reflection skill.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

PROMPTS = Path(__file__).resolve().parent


@cache
def manifest() -> dict[str, dict[str, str]]:
    return json.loads((PROMPTS / "manifest.json").read_text(encoding="utf-8"))["files"]


def load(name: str) -> str:
    """Template text, e.g. load("actions/feedback.md")."""
    return (PROMPTS / name).read_text(encoding="utf-8")


def role_ids() -> list[str]:
    return sorted(Path(name).stem for name in manifest() if name.startswith("roles/"))
