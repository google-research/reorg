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
"""One Stirrup agent session in a sandbox (App. A.1-A.2).

The worker W, for its initial work A_0 and each revision A_r of Eq. 1, and
every stakeholder node run through `run_agent`. A domain runtime supplies the
system prompt, the action prompt and the tools:

    WorkspaceCodeExec(sandbox)          code_exec, started in the sandbox
    view_image()                        view_image on sandbox files (if profile.images)
    finish_tool(expected)               finish(reason, paths), checked against the expected outputs
    ABANDON_TOOL                        abandon_task_finish(reason)
    web_search.worker_web_tools(...)    fetch and search, for the worker only

Default limits (App. A.2): 250 turns, a turns-remaining warning from 50 turns
left, context compaction at 0.7 of the window, 600 s per shell command.
"""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import json
import os
import signal
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import anyio
from pydantic import BaseModel, Field
from stirrup import Agent, TokenUsage, Tool, ToolProvider, ToolResult, ToolUseCountMetadata
from stirrup.core.agent import _SESSION_STATE
from stirrup.tools import ViewImageToolProvider
from stirrup.tools.code_backends.base import CommandResult
from stirrup.tools.code_backends.local import LocalCodeExecToolProvider

from .llm import ModelProfile, stirrup_client


@dataclass
class AgentRun:
    finish: dict | None                 # the finish call's arguments; None if the agent never finished
    finish_type: str | None             # "FinishParams", "AbandonParams" or a runtime's own finish params
    history: list                       # Stirrup message history, as JSON
    usage: dict | None                  # summed token usage: input, answer, reasoning


def run_agent(profile: ModelProfile, *, system_prompt: str, prompt: str, tools: list[Tool | ToolProvider],
              finish_tools: list[Tool], logs: Path, name: str = "worker", max_turns: int = 250,
              turns_remaining_warning_threshold: int = 50, context_summarization_cutoff: float = 0.7,
              input_files: list[Path] | None = None, timeout: float | None = None) -> AgentRun:
    """Run one session to its finish call or turn limit. Model calls are appended to
    logs/model_calls.jsonl; the finish arguments, history and run metadata are written to
    logs/{finish,history,metadata}.json. Stirrup prepends its own base system prompt, which
    lists `input_files` (copied into its workspace) when given; `timeout` bounds the session."""
    logs.mkdir(parents=True, exist_ok=True)
    agent = Agent(client=stirrup_client(profile, logs / "model_calls.jsonl"), name=name, system_prompt=system_prompt,
                  tools=tools, finish_tool=finish_tools, max_turns=max_turns,
                  turns_remaining_warning_threshold=turns_remaining_warning_threshold,
                  context_summarization_cutoff=context_summarization_cutoff)

    async def session() -> tuple[Any, list, dict]:
        # No SIGINT cache: sessions run in worker threads, which cannot install signal handlers.
        async with agent.session(input_files=input_files, output_dir=None, cache_on_interrupt=False) as active:
            return await active.run(prompt)

    finish, history, metadata = asyncio.run(asyncio.wait_for(session(), timeout))
    run = AgentRun(finish=_jsonable(finish) if finish else None,
                   finish_type=type(finish).__name__ if finish else None, history=_jsonable(history),
                   usage=sum(metadata["token_usage"], TokenUsage()).model_dump())
    for stem, payload in (("finish", run.finish), ("history", run.history), ("metadata", _jsonable(metadata))):
        (logs / f"{stem}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return run


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(child) for child in value]
    if isinstance(value, Path):
        return str(value)
    return value


# ---- code execution ----------------------------------------------------------

class WorkspaceCodeExec(LocalCodeExecToolProvider):
    """Stirrup's local code execution inside an existing sandbox, which outlives the session.

    Commands start in the sandbox, and Stirrup rejects commands with absolute paths.
    Paths given to Stirrup's file tools must stay inside the sandbox. Packages the agent
    installs go to a per-sandbox virtual environment over the system site-packages. This
    is trusted local execution, not a security boundary.
    """

    def __init__(self, sandbox: Path, shell_timeout: int = 600) -> None:
        super().__init__(shell_timeout=shell_timeout)
        self._sbox = sandbox

    async def __aenter__(self) -> Tool:
        self._sbox.mkdir(parents=True, exist_ok=True)
        self._temp_dir = self._sbox
        return self.get_code_exec_tool(description=self._description)

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        self._temp_dir = None                       # keep the sandbox: the runtime reads the outputs

    def _resolve_and_validate_path(self, path: str) -> Path:
        if self._temp_dir is None:
            raise RuntimeError("ExecutionEnvironment not started.")
        resolved = Path(path) if Path(path).is_absolute() else self._temp_dir / path
        # Lexical check: staged symlinks may point outside the sandbox, ".." may not.
        if not Path(os.path.normpath(resolved.absolute())).is_relative_to(os.path.normpath(self._temp_dir.absolute())):
            raise ValueError(f"Path is outside execution environment: {path}")
        return resolved

    def _venv(self) -> Path:
        venv = self._temp_dir / ".agent_venv"
        if not (venv / "bin" / "python").exists():
            # Parallel sessions can share a sandbox; create the environment once.
            with (self._temp_dir / ".agent_venv.lock").open("w") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                if not (venv / "bin" / "python").exists():
                    subprocess.run([sys.executable, "-m", "venv", "--system-site-packages", str(venv)],
                                   check=True, capture_output=True, text=True)
        return venv

    def _env(self, venv: Path) -> dict[str, str]:
        env = os.environ.copy()
        env.update(VIRTUAL_ENV=str(venv), UV_PROJECT_ENVIRONMENT=str(venv), UV_LINK_MODE="copy",
                   PIP_REQUIRE_VIRTUALENV="true", PATH=f"{venv / 'bin'}:{env.get('PATH', '')}")
        for key in ("PYTHONHOME", "GEMINI_API_KEY"):    # the agent's shell never sees the API key
            env.pop(key, None)
        return env

    async def run_command(self, cmd: str, *, timeout: int | None = None) -> CommandResult:
        if self._temp_dir is None:
            raise RuntimeError("ExecutionEnvironment not started.")
        timeout = self._shell_timeout if timeout is None else timeout
        if absolute_path_error := self._check_absolute_paths(cmd):
            return absolute_path_error
        try:
            venv = self._venv()
        except subprocess.CalledProcessError as exc:
            return CommandResult(exit_code=1, stdout=exc.stdout or "", stderr=exc.stderr or str(exc),
                                 error_kind="environment_setup_failed")
        process = None
        try:
            with anyio.fail_after(timeout):
                # A new session, so a timeout can kill the whole process group.
                process = await anyio.open_process(["bash", "-c", cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                                   cwd=self._temp_dir, env=self._env(venv), start_new_session=True)
                stdout: list[bytes] = []
                stderr: list[bytes] = []

                async def drain(stream: Any, chunks: list[bytes]) -> None:
                    if stream:
                        chunks.extend([chunk async for chunk in stream])

                async with anyio.create_task_group() as group:
                    group.start_soon(drain, process.stdout, stdout)
                    group.start_soon(drain, process.stderr, stderr)
                await process.wait()
                return CommandResult(exit_code=process.returncode or 0,
                                     stdout=b"".join(stdout).decode("utf-8", errors="replace"),
                                     stderr=b"".join(stderr).decode("utf-8", errors="replace"))
        except asyncio.CancelledError:              # a cancelled session must not leave the command running
            if process:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
            raise
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


def view_image() -> ViewImageToolProvider:
    """view_image on files in the session's sandbox."""
    return ViewImageToolProvider()


# ---- finish tools ------------------------------------------------------------

class FinishParams(BaseModel):
    """Submission finish parameters."""

    reason: Annotated[str, Field(description="Brief summary of what you accomplished.")]
    paths: Annotated[
        list[str],
        Field(description="Relative file paths for deliverable files. Do not submit directories."),
    ]


class AbandonParams(BaseModel):
    """Abandon-task finish parameters."""

    reason: Annotated[str, Field(description="Brief reason the task cannot be completed.")]


def _suffixes(paths: list[str]) -> Counter[str]:
    return Counter(Path(path).suffix.lower() for path in paths if Path(path).suffix)


def finish_tool(expected: list[str]) -> Tool[FinishParams, ToolUseCountMetadata]:
    """finish(reason, paths). Paths must be existing relative files in the sandbox with unique
    basenames; with `expected` outputs, they must also match them in number and extensions."""

    async def executor(params: FinishParams) -> ToolResult[ToolUseCountMetadata]:
        if not params.paths:
            return ToolResult(content="ERROR: finish.paths must contain at least one deliverable file", success=False)
        errors = []
        if expected and len(params.paths) != len(expected):
            errors.append(f"submit exactly {len(expected)} final deliverable file(s), matching the task's expected "
                          "outputs; do not include helper scripts, source files, rendered screenshots, or "
                          "intermediate files unless requested")
        if expected and _suffixes(expected) and _suffixes(params.paths) != _suffixes(expected):
            errors.append("submitted file extensions must match expected deliverables: "
                          f"expected {dict(_suffixes(expected))}, got {dict(_suffixes(params.paths))}")
        exec_env = _SESSION_STATE.get().exec_env
        basenames: set[str] = set()
        duplicates: set[str] = set()
        for path in params.paths:
            if Path(path).is_absolute() or ".." in Path(path).parts or not Path(path).parts:
                errors.append(f"path must be relative and stay inside the task workspace: {path}")
                continue
            if Path(path).name in basenames:
                duplicates.add(Path(path).name)
            basenames.add(Path(path).name)
            if await exec_env.is_directory(path):
                errors.append(f"path is a directory, not a file: {path}")
            elif not await exec_env.file_exists(path):
                errors.append(f"file does not exist: {path}")
        if duplicates:
            errors.append(f"finish paths must have unique basenames for extraction: {sorted(duplicates)}")
        if errors:
            return ToolResult(content="ERROR: " + "; ".join(errors), success=False)
        return ToolResult(content=params.reason, metadata=ToolUseCountMetadata(), success=True)

    return Tool[FinishParams, ToolUseCountMetadata](
        name="finish",
        description=("Submit completed GDPval deliverable files with a brief summary. Paths must be relative files, "
                     "not directories. Submit only final requested deliverables, not helper scripts or intermediate "
                     "files."),
        parameters=FinishParams,
        executor=executor,
    )


async def _abandon(params: AbandonParams) -> ToolResult[ToolUseCountMetadata]:
    return ToolResult(content=params.reason, metadata=ToolUseCountMetadata(), success=True)


ABANDON_TOOL = Tool[AbandonParams, ToolUseCountMetadata](
    name="abandon_task_finish",
    description="Signal that the task genuinely cannot be completed and provide a brief reason.",
    parameters=AbandonParams,
    executor=_abandon,
)
