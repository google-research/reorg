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
"""The judge's `code_exec` tool (App. A.5.2): Stirrup's host-local execution
with a per-attempt virtualenv, a hard budget of calls for the whole judgment,
and a head/tail cap on each result.

This is not a security boundary: commands run on the host. Judge untrusted
artifacts in a disposable VM or container.
"""

from __future__ import annotations

import contextlib
import fcntl
import os
import signal
import subprocess
import sys
from pathlib import Path

import anyio
from stirrup.core.models import Tool, ToolResult, ToolUseCountMetadata
from stirrup.tools.code_backends.base import CodeExecutionParams, CommandResult
from stirrup.tools.code_backends.local import LocalCodeExecToolProvider


class BudgetedLocalCodeExec(LocalCodeExecToolProvider):

    def __init__(self, *, max_uses: int, max_result_chars: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.max_uses = max_uses
        self.max_result_chars = max_result_chars
        self.num_uses = 0

    def _agent_venv(self) -> Path:
        """A venv over the system site packages, so packages the judge installs
        stay inside this attempt."""
        venv_dir = self._temp_dir / ".agent_venv"
        python_path = venv_dir / "bin/python"
        with (self._temp_dir / ".agent_venv.lock").open("w", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            if not python_path.exists():
                subprocess.run([sys.executable, "-m", "venv", "--system-site-packages", str(venv_dir)],
                               check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return venv_dir

    @staticmethod
    def _command_env(venv_dir: Path) -> dict[str, str]:
        env = os.environ.copy()
        env["VIRTUAL_ENV"] = str(venv_dir)
        env["UV_PROJECT_ENVIRONMENT"] = str(venv_dir)
        env["UV_LINK_MODE"] = "copy"
        env["PIP_REQUIRE_VIRTUALENV"] = "true"
        env["PATH"] = f"{venv_dir / 'bin'}:{env.get('PATH', '')}"
        env.pop("PYTHONHOME", None)
        return env

    async def run_command(self, cmd: str, *, timeout: int | None = None) -> CommandResult:
        timeout = self._shell_timeout if timeout is None else timeout
        if not self._check_allowed(cmd):
            return CommandResult(
                exit_code=1,
                stdout="",
                stderr=f"Command not allowed: '{cmd}' does not match any allowed patterns",
                error_kind="command_not_allowed",
                advice="Only commands matching the allowlist patterns are permitted.",
            )
        if absolute_path_error := self._check_absolute_paths(cmd):
            return absolute_path_error
        try:
            venv_dir = self._agent_venv()
        except subprocess.CalledProcessError as exc:
            return CommandResult(exit_code=1, stdout=exc.stdout or "", stderr=exc.stderr or str(exc),
                                 error_kind="environment_setup_failed")
        process = None
        try:
            with anyio.fail_after(timeout):
                # A new session, so a timeout kills the command's whole process group.
                process = await anyio.open_process(
                    ["bash", "-c", cmd],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=self._temp_dir,
                    env=self._command_env(venv_dir),
                    start_new_session=True,
                )
                stdout_chunks: list[bytes] = []
                stderr_chunks: list[bytes] = []

                async def read_stdout() -> None:
                    stdout_chunks.extend([chunk async for chunk in process.stdout])

                async def read_stderr() -> None:
                    stderr_chunks.extend([chunk async for chunk in process.stderr])

                async with anyio.create_task_group() as group:
                    group.start_soon(read_stdout)
                    group.start_soon(read_stderr)
                await process.wait()
                return CommandResult(
                    exit_code=process.returncode or 0,
                    stdout=b"".join(stdout_chunks).decode("utf-8", errors="replace"),
                    stderr=b"".join(stderr_chunks).decode("utf-8", errors="replace"),
                )
        except TimeoutError:
            if process:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                with anyio.move_on_after(5):
                    await process.wait()
            return CommandResult(exit_code=1, stdout="", stderr=f"Command timed out after {timeout} seconds",
                                 error_kind="timeout")
        except Exception as exc:
            return CommandResult(exit_code=1, stdout="", stderr=str(exc), error_kind="execution_error")

    def _cap(self, result: ToolResult[ToolUseCountMetadata]) -> ToolResult[ToolUseCountMetadata]:
        if not isinstance(result.content, str) or len(result.content) <= self.max_result_chars:
            return result
        head_len = max(1, self.max_result_chars // 2)
        tail_len = max(1, self.max_result_chars - head_len)
        content = (
            "<tool_output_truncated>\n"
            f"<original_chars>{len(result.content)}</original_chars>\n"
            f"<kept_chars>{self.max_result_chars}</kept_chars>\n"
            "<instruction>Output was too long. Rerun a narrower command that prints compact evidence only.</instruction>\n"
            f"<head>\n{result.content[:head_len]}\n</head>\n"
            f"<tail>\n{result.content[-tail_len:]}\n</tail>\n"
            "</tool_output_truncated>"
        )
        return ToolResult(content=content, success=result.success, metadata=result.metadata)

    def get_code_exec_tool(self, *, name: str = "code_exec",
                           description: str | None = None) -> Tool[CodeExecutionParams, ToolUseCountMetadata]:
        base = super().get_code_exec_tool(name=name, description=description)

        async def executor(params: CodeExecutionParams) -> ToolResult[ToolUseCountMetadata]:
            # The budget spans the whole attempt; context summarization does not reset it.
            self.num_uses += 1
            if self.num_uses > self.max_uses:
                return ToolResult(
                    content=(
                        "<shell_results>\n<error_kind>code_exec_budget_exhausted</error_kind>\n"
                        f"<details>The code_exec budget of {self.max_uses} calls is exhausted. "
                        "This budget is global for the entire judgment attempt and is not reset by "
                        "context summarization or session continuation. Do not call code_exec again. "
                        "Call finish_judgment now using the evidence already collected.</details>\n"
                        "</shell_results>"
                    ),
                    success=False,
                    metadata=ToolUseCountMetadata(num_uses=1),
                )
            return self._cap(await base.executor(params))

        return Tool[CodeExecutionParams, ToolUseCountMetadata](
            name=base.name,
            description=(base.description or "")
            + f" Hard budget: at most {self.max_uses} code_exec calls; after that, finish the judgment.",
            parameters=base.parameters,
            executor=executor,
        )
