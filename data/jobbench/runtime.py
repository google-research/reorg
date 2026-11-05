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
"""JobBench actions (Sec. 5.1; App. A.4, Table "Benchmark-specific worker and submission settings").

Every action of Eq. 1, the initial work A_0 = W(x), a stakeholder node of
Reflect(x, A_{r-1}, H_{r-1}; s_r) and the revision A_r, is one Stirrup session
(reorg/worker.py) in a staged sandbox:

    task.md                   the request u
    reference_files/          R, symlinked at their dataset paths (main/<occupation>/task<k>/...)
    current_artifact/         A_{r-1}, empty for initial work; the worker writes its deliverables here
    scratch/                  the worker's helper files, never submitted
    evidence/                 guarded downloads kept by earlier rounds (<round>/<node>/...) and
                              behind this round's routed contributions (upstream/<node>/...)
    collaboration_workspace/  contributions and the work chronicle H (reorg/reflection_skill/reflect.py)

JobBench reuses the shared prompts under a short JobBench wrapper:

    # Role / Social Profile               the worker profile (P1) or a stakeholder profile (R1-R10)
    # Frozen Source Action Instruction    the shared action prompt (P1-P3)
    # JobBench Transfer Wrapper           data/jobbench/prompts/<action>.md
    # Original Task                       u
    # Incoming Contributions              the contributions routed to the node
    Work only inside the visible workspace. Begin now.

A Loop-In node runs the protocol of reorg/reflection_skill/loop_in.py in one sandbox; each phase is a
session that ends by writing one file under collaboration_workspace/loop_in/. Only the
worker gets web tools, and only on main tasks (tools.py). An action that succeeds keeps
the evidence files it added, and later actions see them.
"""

from __future__ import annotations

import json
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from reorg.reflection_skill import loop_in
from reorg import prompts
from reorg.reflection_skill.chronicle import STAGED
from reorg.domain import Domain, TaskPacket, load_module
from reorg.files import read_json, sha256_file
from reorg.llm import ModelProfile
from reorg.reflection_skill.reflect import Action, ActionResult
from reorg.reflection_skill.graph import ACTION_PROMPTS, RUN_ROLE_IDS
from reorg.worker import AgentRun, run_agent, view_image

HERE = Path(__file__).resolve().parent
tools = load_module(HERE / "tools.py")
SETTINGS = read_json(HERE / "reorg.json")

SYSTEM_PROMPT = """You are one member of a professional team completing a JobBench task.

Use code_exec for local work. Use only web_search, fetch_web_page, and download_file for internet retrieval; direct network access from code_exec is prohibited. Never search for JobBench, task IDs, task cards, rubrics, benchmark copies, or hidden answers. Treat colleague feedback as advisory and verify it against task.md, visible inputs, public sources, and the current artifact.

Read task inputs from reference_files/. Guarded public-source downloads that
have already been collected for this task are under evidence/.

Work efficiently, validate files before finishing, and call exactly one finish tool when your assigned action is complete."""

# The shared prompt behind each action, and the file a Feedback or Propose node publishes.
SOURCE_PROMPTS = {"initial_work": "worker/initial_work.md", "revise": "worker/worker_revision.md", **ACTION_PROMPTS}
OUTPUTS = {"feedback": "collaboration_workspace/colleague_feedback.md",
           "propose": "collaboration_workspace/improvement_proposal.md"}
# Model-facing: contribution headings name roles by their versioned IDs (RUN_ROLE_IDS).


def _load(name: str) -> str:
    return prompts.load(name).strip()


def wrapper(action: str) -> str:
    return (HERE / "prompts" / f"{action}.md").read_text(encoding="utf-8").strip()


def action_prompt(action: Action, request: str) -> str:
    role = _load(f"roles/{action.role}.md") if action.role else _load("worker/worker_role.md")
    contributions = "\n\n".join(f"## {RUN_ROLE_IDS[c.role]} / {c.action} / {c.status}\n\n{c.text}"
                                for c in action.incoming) or "No colleague contributions are available."
    return "\n\n".join(["# Role / Social Profile\n\n" + role,
                        "# Frozen Source Action Instruction\n\n" + _load(SOURCE_PROMPTS[action.action]),
                        "# JobBench Transfer Wrapper\n\n" + wrapper(action.action),
                        "# Original Task\n\n" + request,
                        "# Incoming Contributions\n\n" + contributions,
                        "Work only inside the visible workspace. Begin now."])


def phase_prompt(role: str, protocol_prompt: str, completion: str) -> str:
    """One Loop-In phase: the speaker's profile, the Loop-In action, the phase protocol, the wrapper."""
    body = "\n\n".join(["# Frozen Source Action\n\n" + _load(ACTION_PROMPTS["loop_in"]),
                        "# Frozen Source Protocol\n\n" + protocol_prompt,
                        "# JobBench Transfer Wrapper\n\n" + wrapper("loop_in"),
                        "# Phase Completion Adapter\n\n" + completion])
    return f"# Role / Social Profile\n\n{_load(f'roles/{role}.md')}\n\n{body}"


def _digests(root: Path) -> dict[Path, str]:
    return {path.relative_to(root): sha256_file(path) for path in root.rglob("*") if path.is_file()}


@dataclass
class JobBenchRuntime:
    packet: TaskPacket
    data: Path                                  # <data root>/jobbench; reference paths are relative to it
    profile: ModelProfile
    web: bool                                   # guarded retrieval for the worker (main tasks)
    settings: dict

    # Sandboxes sit at <task>/<round>/nodes/<node>/attempt_NN/sandbox (reorg/reflection_skill/reflect.py), so
    # sandbox.parents[2] holds this round's nodes and sandbox.parents[3] is the round.

    def stage(self, sandbox: Path, artifact: Path | None) -> None:
        sandbox.mkdir(parents=True)
        (sandbox / "task.md").write_text(self.packet.request + "\n", encoding="utf-8", newline="\n")
        (sandbox / "reference_files").mkdir()
        for path in self.packet.references:
            link = sandbox / "reference_files" / path.relative_to(self.data)
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(path.resolve())
        if artifact:
            shutil.copytree(artifact, sandbox / "current_artifact")
        else:
            (sandbox / "current_artifact").mkdir()
        (sandbox / "scratch").mkdir()
        (sandbox / "evidence").mkdir()
        round_dir = sandbox.parents[3]              # evidence kept by every action of earlier rounds
        for kept in sorted(round_dir.parent.glob("*/nodes/*/attempt_*/evidence")):
            if kept.parents[3] != round_dir:
                shutil.copytree(kept, sandbox / "evidence" / kept.parents[3].name / kept.parents[1].name,
                                dirs_exist_ok=True)

    def run(self, action: Action, sandbox: Path, logs: Path) -> ActionResult:
        evidence = sandbox / "evidence"
        for contribution in action.incoming:        # evidence behind each published contribution routed here
            if contribution.status == "available":
                for kept in sandbox.parents[2].glob(f"{contribution.vertex_id}/attempt_*/evidence"):
                    shutil.copytree(kept, evidence / "upstream" / contribution.vertex_id, dirs_exist_ok=True)
        before = _digests(evidence)
        result = self._loop_in(action, sandbox, logs) if action.action == "loop_in" else self._act(action, sandbox, logs)
        if result.status == "ok":                  # keep only what the action added or changed
            for relative, digest in _digests(evidence).items():
                if before.get(relative) != digest:
                    target = logs.parent / "evidence" / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(evidence / relative, target)
        return result

    def session(self, name: str, prompt: str, sandbox: Path, logs: Path, outputs: list[str] | None, *, web: bool,
                max_turns: int) -> AgentRun:
        """One Stirrup session; `outputs` are the files a stakeholder or Loop-In phase must write."""
        s = self.settings
        session_tools = [tools.GuardedCodeExec(sandbox, log_path=logs / "code_exec.jsonl", outputs=outputs,
                                               pre_output_limit=s["stakeholder_pre_output_calls"],
                                               post_output_limit=s["stakeholder_post_output_calls"] if outputs else None,
                                               shell_timeout=s["shell_timeout_seconds"])]
        if web:
            budget = tools.RetrievalBudget(search=s["search_limit"], fetch=s["fetch_limit"],
                                           download=s["download_limit"], byte_limit=s["download_byte_limit"],
                                           log_path=logs / "retrieval.jsonl")
            session_tools += tools.retrieval_tools(task_id=self.packet.task_id, task_text=self.packet.request,
                                                   budget=budget, settings=s)
        if self.profile.images:
            session_tools.append(view_image())
        return run_agent(self.profile, system_prompt=SYSTEM_PROMPT, prompt=prompt, tools=session_tools,
                         finish_tools=[tools.finish_tool(outputs), tools.ABANDON_TOOL], logs=logs,
                         name=f"jobbench_{name}", max_turns=max_turns,
                         turns_remaining_warning_threshold=s["turn_warning_threshold"],
                         context_summarization_cutoff=s["context_summarization_cutoff"])

    def _act(self, action: Action, sandbox: Path, logs: Path) -> ActionResult:
        worker = action.role is None
        output = OUTPUTS.get(action.action)
        run = self.session(action.vertex_id, action_prompt(action, self.packet.request), sandbox, logs,
                           [output] if output else None, web=worker and self.web, max_turns=self.settings["max_turns"])
        if run.finish and not run.finish_type.startswith("Abandon"):
            status = "ok"
        else:                                   # a worker out of turns keeps its latest artifact
            status = "turn_limit_best_available" if worker and not run.finish else "model_failure"
        text = (sandbox / output).read_text(encoding="utf-8") if output and (sandbox / output).is_file() else ""
        return ActionResult(status, text=text, artifact=sandbox / "current_artifact" if worker else None,
                            logs={"logs_dir": str(logs), "finish_type": run.finish_type}, usage=run.usage)

    def _loop_in(self, action: Action, sandbox: Path, logs: Path) -> ActionResult:
        driver = LoopInDriver(self, sandbox, logs / "loop_in")
        result = loop_in.run(action.role, driver)
        usage = Counter()
        for run in driver.runs:
            usage.update(run.usage)
        return ActionResult("ok" if result.status == "ok" else "protocol_failure", text=loop_in.outcome(result),
                            logs={"logs_dir": str(logs), "phase": result.phase, "error": result.error},
                            usage=dict(usage))


class LoopInDriver:
    """loop_in.Driver for JobBench: each phase is a session that must write one file, then finish."""

    def __init__(self, runtime: JobBenchRuntime, sandbox: Path, logs: Path) -> None:
        self.runtime = runtime
        self.sandbox = sandbox
        self.logs = logs
        self.chronicle = (sandbox / STAGED).read_bytes()
        self.runs: list[AgentRun] = []

    def select(self, initiator: str, protocol_prompt: str) -> tuple[str, str]:
        output = "collaboration_workspace/loop_in/selection.json"
        choice = json.loads(self._phase("select", initiator, protocol_prompt, output,
                                        "Write a JSON object with keys `partner_display_name` and `opening_message` "
                                        f"to `{output}`, then call finish with that path."))
        return str(choice["partner_display_name"]), str(choice["opening_message"])

    def turn(self, speaker: str, protocol_prompt: str, label: str) -> str:
        participant, round_index = label.rsplit("_", 1)
        output = f"collaboration_workspace/loop_in/{participant}_{int(round_index):03d}.json"
        reply = self._phase(label, speaker, protocol_prompt, output,
                            f"Write the `send_message` content as JSON key `message` to `{output}`, then finish.")
        return str(json.loads(reply)["message"])

    def publish(self, initiator: str, protocol_prompt: str) -> str:
        output = "collaboration_workspace/loop_in/summary.md"
        return self._phase("publish", initiator, protocol_prompt, output,
                           f"Write the required handoff to `{output}`, then finish with that path.").strip()

    def _phase(self, label: str, role: str, protocol_prompt: str, output: str, completion: str) -> str:
        target = self.sandbox / output
        target.parent.mkdir(parents=True, exist_ok=True)
        target.unlink(missing_ok=True)
        (self.sandbox / STAGED).write_bytes(self.chronicle)     # every phase sees the node's pre-action chronicle
        run = self.runtime.session(f"loop_in_{label}", phase_prompt(role, protocol_prompt, completion), self.sandbox,
                                   self.logs / f"{len(self.runs) + 1:02d}_{label}", [output], web=False,
                                   max_turns=self.runtime.settings["loop_in_phase_max_turns"])
        self.runs.append(run)
        if not run.finish or not target.is_file():
            raise ValueError(f"Loop In phase {label} did not publish {output}")
        return target.read_text(encoding="utf-8")


def make_runtime(domain: Domain, task_id: str, profile: ModelProfile, config: dict) -> JobBenchRuntime:
    """The runtime of one task. `config` is the inference config; JobBench's settings are in reorg.json."""
    return JobBenchRuntime(domain.load_task(task_id), domain.data, profile,
                           web=task_id.split("/")[0] in SETTINGS["retrieval_splits"], settings=SETTINGS)


def submission(domain: Domain, task_id: str, artifact_dir: Path, out_dir: Path) -> list[str]:
    """Copy the judged files to out_dir: every file of the final artifact.

    The worker writes the deliverables it declares at finish to current_artifact/ and its
    helpers to scratch/, which is not part of the artifact; the declared paths themselves
    are checked to exist but do not select files."""
    shutil.rmtree(out_dir, ignore_errors=True)
    shutil.copytree(artifact_dir, out_dir)
    return sorted(path.relative_to(out_dir).as_posix() for path in out_dir.rglob("*") if path.is_file())
