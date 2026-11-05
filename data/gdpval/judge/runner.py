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
"""One judge's inspection of one comparison (App. A.5.2): a Stirrup session
with code execution and image viewing, no web, ended by `finish_judgment`.

An attempt that ends without a judgment, times out, or fails in the agent is
retried once in a fresh workspace; provider failures are not retried here
(the clients already back off). Each judge judges each pair once: there are no
A/B swap audits.

Layout per member and comparison:

    <out>/<member>/<attempt_id>/summary.json
    <out>/<member>/<attempt_id>/try_<k>/{sandbox/, logs/{model_calls.jsonl, history.json, finish.json, ...}}
"""

from __future__ import annotations

import asyncio
import contextlib
import shutil
import signal
import time
from pathlib import Path
from typing import Any

from reorg.files import write_json

from . import sandbox
from .local_exec import BudgetedLocalCodeExec
from .prompting import JUDGE_SYSTEM_PROMPT, build_judge_prompt, finish_judgment_tool
from .providers import make_client

RETRIED = {"no_finish", "timeout", "agent_error"}
# Provider and network failures are infrastructure, not judge behavior.
INFRA_MARKERS = ("RateLimit", "HTTPStatusError", "ConnectError", "ReadTimeout", "TimeoutException", "API key",
                 "RESOURCE_EXHAUSTED", "Quota exceeded", "APIConnectionError", "APITimeoutError",
                 "InternalServerError")


class AttemptTimeout(TimeoutError):
    pass


@contextlib.contextmanager
def wallclock_timeout(seconds: float):
    """SIGALRM bound on a whole attempt; judges run in worker processes' main threads."""
    def expire(_signum, _frame):
        raise AttemptTimeout(f"worker exceeded timeout_seconds={seconds:g}")

    previous = signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(key): jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(child) for child in value]
    if isinstance(value, Path):
        return str(value)
    return value


async def _session(prompt: str, workspace: Path, logs: Path, exec_dir: Path, profile: dict, panel: dict):
    from stirrup import Agent
    from stirrup.tools import ViewImageToolProvider

    code_exec = BudgetedLocalCodeExec(temp_base_dir=exec_dir, shell_timeout=panel["shell_timeout_seconds"],
                                      max_uses=panel["code_exec_max_uses"],
                                      max_result_chars=panel["code_exec_result_max_chars"])
    agent = Agent(
        client=make_client(profile, logs / "model_calls.jsonl"),
        name="gdpval_stirrup_judge",
        system_prompt=JUDGE_SYSTEM_PROMPT,
        tools=[code_exec, ViewImageToolProvider()],
        finish_tool=finish_judgment_tool(),
        max_turns=panel["max_turns"],
        turns_remaining_warning_threshold=panel["turn_warning_threshold"],
        context_summarization_cutoff=panel["context_summarization_cutoff"],
    )
    inputs = sorted(workspace.iterdir(), key=lambda path: path.name)
    async with agent.session(input_files=inputs, cache_on_interrupt=False) as session:
        finish, history, _ = await session.run(prompt)
    return jsonable(finish) if finish else None, jsonable(history), code_exec.num_uses


def _try(plan: sandbox.Plan, task: dict, data: Path, profile: dict, panel: dict, try_dir: Path) -> dict:
    logs = try_dir / "logs"
    logs.mkdir(parents=True)
    row: dict[str, Any] = {"task_id": plan.task_id, "attempt_id": plan.attempt_id, "mapping": plan.mapping}
    started = time.time()
    try:
        prompt = build_judge_prompt(panel["code_exec_max_uses"])
        sandbox.stage(plan, task, data, try_dir / "sandbox", prompt)
        with (logs / "stirrup_stdout.log").open("w", encoding="utf-8") as out, \
                (logs / "stirrup_stderr.log").open("w", encoding="utf-8") as err, \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            finish, history, uses = asyncio.run(
                _session(prompt, try_dir / "sandbox", logs, try_dir / "local_exec_tmp", profile, panel))
        write_json(logs / "history.json", history)
        write_json(logs / "finish.json", finish)
        row["code_exec_uses"] = uses
        if finish is None:
            row["status"] = "no_finish"
        else:
            row.update(status="ok", preference=finish["preference"], analysis=finish["analysis"],
                       response_a_raw_points=finish["response_a_raw_points"],
                       response_b_raw_points=finish["response_b_raw_points"])
    except AttemptTimeout as exc:
        row.update(status="timeout", error=repr(exc))
    except Exception as exc:
        infra = any(marker in repr(exc) for marker in INFRA_MARKERS)
        row.update(status="infra_error" if infra else "agent_error", error=repr(exc))
    row["duration_seconds"] = round(time.time() - started, 3)
    return row


def judge(member: str, profile: dict, panel: dict, plan: sandbox.Plan, task: dict, data: Path, out: Path) -> dict:
    """One member's judgment of one comparison; returns its results row."""
    attempt_dir = out / member / plan.attempt_id
    shutil.rmtree(attempt_dir, ignore_errors=True)
    for retry in range(panel["official_retries"] + 1):
        with wallclock_timeout(panel["attempt_timeout_seconds"]):
            row = _try(plan, task, data, profile, panel, attempt_dir / f"try_{retry}")
        if row["status"] not in RETRIED:
            break
    row.update(member=member, retries=retry)
    write_json(attempt_dir / "summary.json", row)
    return row
