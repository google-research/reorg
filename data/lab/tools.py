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
"""LAB's Stirrup tools (App. A.2): a finish tool bound to the exact output files
of an action, and code execution under the finalization guard.

The guard bounds the actions it wraps (initial work, Feedback, Propose and every
Loop-In phase; never the worker's revision): after `pre_output_calls` code_exec
calls without the required outputs it refuses commands that do not touch them,
and once the outputs exist and are structurally valid it allows
`post_output_calls` more calls before demanding `finish`. Its messages are
model-facing.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, Field
from stirrup import Tool, ToolResult, ToolUseCountMetadata
from stirrup.core.agent import _SESSION_STATE
from stirrup.tools.code_backends.base import CommandResult

from reorg.worker import WorkspaceCodeExec

OFFICE_SUFFIXES = {".docx", ".xlsx", ".pptx"}


def valid_outputs(root: Path, paths: list[str]) -> bool:
    """Every path is a nonempty file; Office files are ZIP containers and PDFs carry a PDF header."""
    for relative in paths:
        path = root / relative
        if not path.is_file() or path.stat().st_size == 0:
            return False
        if path.suffix.lower() in OFFICE_SUFFIXES and not zipfile.is_zipfile(path):
            return False
        if path.suffix.lower() == ".pdf" and not path.read_bytes().startswith(b"%PDF-"):
            return False
    return True


class LabFinishParams(BaseModel):
    # No docstring: a class docstring would enter the tool schema the model sees.
    reason: Annotated[str, Field(description="Brief summary of the completed work.")]
    paths: Annotated[list[str], Field(description="Exact relative paths to every required output file.")]


def finish_tool(expected: list[str]) -> Tool[LabFinishParams, ToolUseCountMetadata]:
    """finish(reason, paths): paths must be exactly `expected`, each an existing file."""

    async def executor(params: LabFinishParams) -> ToolResult[ToolUseCountMetadata]:
        if sorted(params.paths) != sorted(expected):
            return ToolResult(content=f"ERROR: submit exactly these paths: {expected}", success=False)
        exec_env = _SESSION_STATE.get().exec_env
        errors = []
        for text in params.paths:
            path = Path(text)
            if path.is_absolute() or ".." in path.parts or not path.parts:
                errors.append(f"path must stay inside the workspace: {text}")
            elif await exec_env.is_directory(text):
                errors.append(f"path is a directory: {text}")
            elif not await exec_env.file_exists(text):
                errors.append(f"file does not exist: {text}")
        if errors:
            return ToolResult(content="ERROR: " + "; ".join(errors), success=False)
        return ToolResult(content=params.reason, success=True, metadata=ToolUseCountMetadata())

    return Tool(name="finish", description="Submit exactly the required LAB action files after validating them.",
                parameters=LabFinishParams, executor=executor)


class FinalizationGuard(WorkspaceCodeExec):
    """code_exec that forces the transition from analysis to outputs to finish."""

    def __init__(self, sandbox: Path, expected: list[str], *, pre_output_calls: int, post_output_calls: int,
                 shell_timeout: int) -> None:
        super().__init__(sandbox, shell_timeout)
        self.expected = list(expected)            # a list: its repr appears in the guard's messages
        self.pre_output_calls = pre_output_calls
        self.post_output_calls = post_output_calls
        self.calls = 0
        self.calls_after_output = 0

    async def run_command(self, cmd: str, *, timeout: int | None = None) -> CommandResult:
        self.calls += 1
        if not valid_outputs(self._sandbox, self.expected):
            mentions_output = any(path in cmd or Path(path).name in cmd for path in self.expected)
            if self.calls > self.pre_output_calls and not mentions_output:
                return CommandResult(exit_code=1, stdout="", error_kind="finish_required", stderr=(
                    "The analysis budget is exhausted. Create every required output now in one "
                    f"code_exec call: {self.expected}. Then call finish."))
        else:
            self.calls_after_output += 1
            if self.calls_after_output > self.post_output_calls:
                return CommandResult(exit_code=1, stdout="", error_kind="finish_required", stderr=(
                    f"Every required output already exists: {self.expected}. "
                    "The validation budget is exhausted. Call finish now with those exact paths."))

        result = await super().run_command(cmd, timeout=timeout)
        if valid_outputs(self._sandbox, self.expected):
            remaining = max(0, self.post_output_calls - self.calls_after_output)
            reminder = ("\n<lab_finalization_reminder>All required outputs exist. "
                        f"Use at most {remaining} more code_exec calls for concrete validation or fixes, "
                        f"then call finish with exactly {self.expected}.</lab_finalization_reminder>")
        elif self.calls >= max(1, self.pre_output_calls - 2):
            reminder = ("\n<lab_finalization_reminder>Stop broad inspection. Create every required output now: "
                        f"{self.expected}, then call finish.</lab_finalization_reminder>")
        else:
            return result
        return CommandResult(exit_code=result.exit_code, stdout=result.stdout + reminder, stderr=result.stderr,
                             error_kind=result.error_kind, advice=result.advice)
