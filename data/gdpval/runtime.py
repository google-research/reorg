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
"""GDPval actions (Sec. 5.1; App. A.1, "Skill Execution and Information Flow"; App. A.4).

Every action of Eq. 1 (the worker's initial work A_0 and revisions A_r, and each
stakeholder node of s_r) is one Stirrup session in its own sandbox:

    task.md                    the request u
    reference_files/           R, symlinked at their dataset paths without the leading reference_files/
    current_artifact/          a copy of A_{r-1} (empty before A_0); the worker edits it in place
    scratch/                   temporary files, never submitted
    collaboration_workspace/   H_{r-1} and the routed contributions, staged by reorg.reflection_skill.reflect

One system prompt serves every action; the action prompt wraps the role profile and
the action prompt in a fixed workspace preamble with the declared deliverables, the
finish paths, the reference list and the task. The declared deliverables are the
dataset paths of the expert deliverables (deliverable_files/<hash>/<name>); the
worker finishes with current_artifact/<name>. The worker also gets web tools,
stakeholders only local ones.

Loop-In runs four sessions in one sandbox: partner selection, a partner turn, an
initiator turn and the initiator's publication. Each phase ends with its own
completion tool, and an invalid completion gets one corrective retry.
"""

from __future__ import annotations

import shutil
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Callable

from pydantic import BaseModel, Field
from stirrup import Tool, ToolResult, ToolUseCountMetadata

from reorg.reflection_skill import loop_in
from reorg import prompts
from reorg.domain import Domain, TaskPacket
from reorg.files import read_json
from reorg.llm import ModelProfile
from reorg.reflection_skill.reflect import Action, ActionResult
from reorg.reflection_skill.graph import ACTION_PROMPTS
from reorg.web_search import worker_web_tools
from reorg.worker import ABANDON_TOOL, AgentRun, WorkspaceCodeExec, finish_tool, run_agent, view_image

HERE = Path(__file__).resolve().parent

# System prompts: ACTION_SYSTEM_PROMPT alone for every action (GDPval has no finalization
# guard), and the initiator or partner prompt for Loop-In phases.
ACTION_SYSTEM_PROMPT = """You are collaborating on one step of a professional task.

Use the provided local workspace and tools to complete exactly the requested step. You cannot interact with the user. Make reasonable assumptions, keep command output compact, and call `finish` as soon as the required output exists and has passed one quick sanity check.

If the action genuinely cannot be completed because required inputs are missing or incoherent, call `abandon_task_finish` with a brief reason. Do not use it to escape difficulty."""

INITIATOR_SYSTEM_PROMPT = """You are the initiating colleague in a bounded professional consultation.

Follow the current phase instructions and use exactly the requested phase-completion tool. When sending a discussion message, advance multiple material threads when useful. You cannot interact with the user. Use only the visible local workspace and do not use web retrieval. Do not edit the task-owner artifact. The protocol runner, not either participant, controls phase transitions."""

PARTNER_SYSTEM_PROMPT = """You are the selected colleague in a bounded professional consultation.

Use your private social-role instructions and the visible local workspace. You know the initiator only by the displayed social-role name provided in the discussion prompt. Send exactly one substantive, potentially multi-topic message with `send_message`. You cannot interact with the user. Do not use web retrieval or edit the task-owner artifact."""

WORKER_PROMPTS = {"initial_work": "worker/initial_work.md", "revise": "worker/worker_revision.md"}
OUTPUTS = {"feedback": "collaboration_workspace/colleague_feedback.md",
           "propose": "collaboration_workspace/improvement_proposal.md",
           "loop_in": "collaboration_workspace/loop_in_outcome.md"}
# Sections a stakeholder output must fill to be published.
REQUIRED = {"feedback": ("# Handoff Summary",),
            "propose": ("# Handoff Summary", "# Perspective and Needs", "# Identified Gaps", "# Prioritized Proposals")}


# ---- Loop-In phase tools -----------------------------------------------------
# No docstrings: pydantic would put them into the tool schemas the model sees.

class PartnerSelectionParams(BaseModel):
    partner_display_name: str = Field(description="One exact display name from the eligible partner list.")
    opening_message: str = Field(
        description="A compact, potentially multi-topic opening agenda to send to the selected colleague.")


class TeamMessageParams(BaseModel):
    message: str = Field(
        description="One substantive, potentially multi-topic discussion message to send to the other colleague.")


async def _selected(params: PartnerSelectionParams) -> ToolResult[ToolUseCountMetadata]:
    return ToolResult(content=params.opening_message, metadata=ToolUseCountMetadata(), success=True)


async def _sent(params: TeamMessageParams) -> ToolResult[ToolUseCountMetadata]:
    return ToolResult(content=params.message, metadata=ToolUseCountMetadata(), success=True)


SELECT_TOOL = Tool[PartnerSelectionParams, ToolUseCountMetadata](
    name="select_partner",
    description="Select one eligible colleague by exact display name and send a compact opening agenda.",
    parameters=PartnerSelectionParams,
    executor=_selected,
)
MESSAGE_TOOL = Tool[TeamMessageParams, ToolUseCountMetadata](
    name="send_message",
    description="Send exactly one substantive, potentially multi-topic message to the other colleague.",
    parameters=TeamMessageParams,
    executor=_sent,
)


# ---- prompts -----------------------------------------------------------------

def finish_paths(action: Action, packet: TaskPacket) -> list[str]:
    """The paths an action finishes with: the worker's deliverables, or a stakeholder's output file."""
    if action.role:
        return [OUTPUTS[action.action]]
    return [f"current_artifact/{PurePosixPath(path).name}" for path in packet.deliverables]


def action_prompt(action: Action, packet: TaskPacket, references: list[str]) -> str:
    """The action prompt: workspace preamble, role profile, action instruction, references and task."""
    role = prompts.load(f"roles/{action.role}.md" if action.role else "worker/worker_role.md").strip()
    instruction = prompts.load(WORKER_PROMPTS.get(action.action) or ACTION_PROMPTS[action.action]).strip()
    deliverables = "\n".join(f"- `{path}`" for path in packet.deliverables) or \
        "No fixed deliverable filenames were declared."
    if action.action == "loop_in":
        finish = ("- Follow the teaming runner's phase-specific tool instructions. After the discussion, "
                  f"the initiator calls `finish` with `{OUTPUTS['loop_in']}`.")
    else:
        finish = "\n".join(f"- `{path}`" for path in finish_paths(action, packet)) or \
            "- Submit the final action output file(s) you created."
    listed = "\n".join(f"- `reference_files/{path}`" for path in references) or "This task has no reference files."
    return f"""## Project Workspace

You are working in a local Linux project workspace. Every `code_exec` command starts from this workspace. Use relative paths only.

Visible workspace:

- `task.md`: the original task.
- `reference_files/`: task reference files. This may contain symlinks. Use `find -L`, `rg --follow`, and `ls -laL` when inspecting it.
- `current_artifact/`: the current task-owner artifact workspace.
- `collaboration_workspace/work_chronicle.md`: current collaboration status, working memory, and your next step.
- `collaboration_workspace/colleague_contributions.md`: index of colleague contributions available for this step.
- `collaboration_workspace/contributions/`: colleague feedback or debate records referenced by the contribution index.
- `scratch/`: optional temporary files. Nothing in `scratch/` is submitted or passed forward as the artifact.

Do not inspect hidden metadata, controller files, graph specs, judge outputs, expert deliverables, or files outside this workspace.

If you have concise context that should carry forward, append exactly one `## Work Chronicle Update` section to `collaboration_workspace/work_chronicle.md`. Do not add any other new headings before it, and do not rewrite earlier chronicle sections.

Available colleague contribution count: {len(action.incoming)}

Expected final deliverables declared for this task:

{deliverables}

For this action, call `finish` with these relative path(s) when available:

{finish}

## Role / Social Profile

{role}

## Action Instruction

{instruction}

## Reference Files

{listed}

## Original Task

{packet.request}

Begin the action now.
"""


def _filled(text: str, heading: str) -> bool:
    """Whether `heading` has a nonempty body before the next heading of the same or a higher level."""
    lines = [line.strip() for line in text.splitlines()]
    if heading not in lines:
        return False
    start, level = lines.index(heading) + 1, len(heading) - len(heading.lstrip("#"))
    end = next((i for i in range(start, len(lines))
                if lines[i].startswith("#") and len(lines[i]) - len(lines[i].lstrip("#")) <= level), len(lines))
    return any(lines[start:end])


def _status(run: AgentRun, valid: bool) -> str:
    """`valid`: the action's output passes its check. At the turn limit, a valid output is the
    best available one."""
    if run.finish_type == "FinishParams":
        return "ok" if valid else "protocol_failure"
    return "turn_limit_best_available" if run.finish is None and valid else "model_failure"


# ---- runtime -----------------------------------------------------------------

class GDPvalRuntime:
    def __init__(self, packet: TaskPacket, data: Path, profile: ModelProfile, settings: dict) -> None:
        self.packet, self.profile, self.settings = packet, profile, settings
        # sandbox path under reference_files/ -> data file
        self.references = {path.relative_to(data / "reference_files"): path for path in packet.references}

    def stage(self, sandbox: Path, artifact: Path | None) -> None:
        for name in ("reference_files", "current_artifact", "scratch"):
            (sandbox / name).mkdir(parents=True, exist_ok=True)
        (sandbox / "task.md").write_text(self.packet.request + "\n", encoding="utf-8", newline="\n")
        for relative, source in self.references.items():
            (sandbox / "reference_files" / relative).parent.mkdir(parents=True, exist_ok=True)
            (sandbox / "reference_files" / relative).symlink_to(source.resolve())
        if artifact:
            shutil.copytree(artifact, sandbox / "current_artifact", dirs_exist_ok=True)

    def run(self, action: Action, sandbox: Path, logs: Path) -> ActionResult:
        # Listed in filesystem walk order.
        references = [path.relative_to(sandbox / "reference_files").as_posix()
                      for path in (sandbox / "reference_files").rglob("*") if path.is_file()]
        prompt = action_prompt(action, self.packet, references)
        if action.role:                        # stakeholder outputs start as empty files
            (sandbox / OUTPUTS[action.action]).write_text("", encoding="utf-8")
        if action.action == "loop_in":
            driver = LoopInDriver(self, action.role, prompt, sandbox, logs)
            result = loop_in.run(action.role, driver)
            if result.status != "ok":                  # the core publishes a "missing" contribution
                return ActionResult("protocol_failure", logs={"phase": result.phase, "error": result.error},
                                    usage=dict(driver.usage))
            return ActionResult("ok", text=outcome(result),
                                logs={"phase": result.phase, "partner": result.partner, "error": result.error},
                                usage=dict(driver.usage))

        expected = finish_paths(action, self.packet)
        run = self.session(action.vertex_id, ACTION_SYSTEM_PROMPT, prompt, finish_tool(expected), sandbox, logs,
                           web=self.settings["web_search"]["stakeholders" if action.role else "worker"],
                           max_turns=self.settings["max_turns"])
        if action.role is None:
            # A finish passed the finish tool's checks; at the turn limit, the snapshot counts
            # only if every expected deliverable is a nonempty file.
            snapshot = all((sandbox / path).is_file() and (sandbox / path).stat().st_size for path in expected)
            return ActionResult(_status(run, run.finish is not None or snapshot), artifact=sandbox / "current_artifact",
                                logs={"finish_type": run.finish_type}, usage=run.usage)
        text = (sandbox / OUTPUTS[action.action]).read_text(encoding="utf-8", errors="replace")
        return ActionResult(_status(run, all(_filled(text, heading) for heading in REQUIRED[action.action])),
                            text=text, logs={"finish_type": run.finish_type}, usage=run.usage)

    def session(self, name: str, system_prompt: str, prompt: str, finish: Tool, sandbox: Path, logs: Path, *,
                web: bool, max_turns: int) -> AgentRun:
        """One Stirrup session in `sandbox`: code_exec, web tools if `web`, view_image if the model takes images."""
        tools = [WorkspaceCodeExec(sandbox, self.settings["shell_timeout"])]
        if web:
            tools += worker_web_tools(self.profile, task_text=prompt, log_path=logs / "search_calls.jsonl")
        if self.profile.images:
            tools.append(view_image())
        return run_agent(self.profile, system_prompt=system_prompt, prompt=prompt, tools=tools,
                         finish_tools=[finish, ABANDON_TOOL], logs=logs, name=name, max_turns=max_turns,
                         turns_remaining_warning_threshold=self.settings["turns_remaining_warning_threshold"],
                         context_summarization_cutoff=self.settings["context_summarization_cutoff"])


def outcome(result: loop_in.LoopInResult) -> str:
    """The published Loop-In contribution on GDPval: the initiator's synthesis, then the
    discussion as quoted blocks (JobBench and LAB use loop_in.outcome)."""
    blocks = ["# Discussion Transcript"]
    for message in result.messages:
        stage = "Opening" if message.round_index == 0 else f"Round {message.round_index}"
        quoted = [f"> {line}" if line else ">" for line in message.text.splitlines()]
        blocks += ["", f"## {stage} — {message.display_name}", "", *quoted]
    return f"{(result.summary or '').rstrip()}\n\n" + "\n".join(blocks) + "\n"


class LoopInDriver:
    """loop_in.Driver: each phase is a fresh session in the action's sandbox.

    Selection and publication prompts extend the Loop-In action prompt; discussion
    turns append the original task.
    """

    def __init__(self, runtime: GDPvalRuntime, initiator: str, prompt: str, sandbox: Path, logs: Path) -> None:
        self.runtime, self.initiator, self.prompt, self.sandbox, self.logs = runtime, initiator, prompt, sandbox, logs
        self.usage: Counter = Counter()

    def select(self, initiator: str, protocol_prompt: str) -> tuple[str, str]:
        names = loop_in.partners(initiator)

        def check(run: AgentRun) -> str | None:
            if run.finish_type != "PartnerSelectionParams":
                return "call select_partner, not another completion tool"
            if run.finish["partner_display_name"].strip() not in names:
                return f"partner_display_name must exactly match an eligible display name: {sorted(names)}"
            return None if run.finish["opening_message"].strip() else "opening_message must be nonempty"

        run = self._phase("select", initiator, f"{self.prompt}\n\n{protocol_prompt}", SELECT_TOOL, check)
        return run.finish["partner_display_name"], run.finish["opening_message"]

    def turn(self, speaker: str, protocol_prompt: str, label: str) -> str:
        def check(run: AgentRun) -> str | None:
            if run.finish_type != "TeamMessageParams":
                return "call send_message, not another completion tool"
            return None if run.finish["message"].strip() else "discussion message must be nonempty"

        prompt = f"{protocol_prompt}\n\n## Original Task\n\n{self.runtime.packet.request}"
        return self._phase(label, speaker, prompt, MESSAGE_TOOL, check).finish["message"]

    def publish(self, initiator: str, protocol_prompt: str) -> str:
        output = self.sandbox / OUTPUTS["loop_in"]

        def check(run: AgentRun) -> str | None:
            if run.finish_type != "FinishParams":
                return "call finish with the teaming outcome path"
            if run.finish["paths"] != [OUTPUTS["loop_in"]]:
                return f"finish paths must be exactly [{OUTPUTS['loop_in']!r}]"
            text = output.read_text(encoding="utf-8", errors="replace")
            if not text.strip():
                return f"empty loop_in summary: {output}"
            if not _filled(text, "# Handoff Summary"):
                return "loop_in summary requires a nonempty # Handoff Summary section"
            if [line.strip() for line in text.splitlines() if line.startswith("# ")] != ["# Handoff Summary"]:
                return "loop_in summary must contain exactly one top-level section: # Handoff Summary"
            return None

        self._phase("publish", initiator, f"{self.prompt}\n\n{protocol_prompt}", finish_tool([OUTPUTS["loop_in"]]),
                    check)
        return output.read_text(encoding="utf-8")

    def _phase(self, label: str, speaker: str, prompt: str, tool: Tool,
               check: Callable[[AgentRun], str | None]) -> AgentRun:
        """One phase session; an invalid completion or a failed session is retried once with the error."""
        system = INITIATOR_SYSTEM_PROMPT if speaker == self.initiator else PARTNER_SYSTEM_PROMPT
        system = f"{system}\n\n{prompts.load(f'roles/{speaker}.md').strip()}"
        error = None
        for attempt in (1, 2):
            text = prompt if attempt == 1 else (
                f"{prompt}\n\n## Corrective Retry\n\n"
                "Your prior phase completion was invalid. Correct it now without restarting the task. "
                f"Problem: {error}. Use exactly the completion tool required by this phase.")
            try:
                run = self.runtime.session(
                    f"loop_in_{label}", system, text, tool, self.sandbox,
                    self.logs / (label if attempt == 1 else f"{label}_retry1"),
                    web=self.runtime.settings["web_search"]["stakeholders"],
                    max_turns=self.runtime.settings["loop_in_phase_max_turns"])
                self.usage.update(run.usage)
                error = check(run)
            except Exception as exc:
                error = repr(exc)
            if error is None:
                return run
        raise ValueError(error)


# ---- domain hooks --------------------------------------------------------------

def make_runtime(domain: Domain, task_id: str, profile: ModelProfile, config: dict) -> GDPvalRuntime:
    """The action settings come from data/gdpval/reorg.json; `config` (inference) is not needed here."""
    return GDPvalRuntime(domain.load_task(task_id), domain.data, profile, read_json(HERE / "reorg.json"))


def submission(domain: Domain, task_id: str, artifact_dir: Path, out_dir: Path) -> list[str]:
    """Every artifact file is judged: all of current_artifact/ except hidden files."""
    files = sorted(path.relative_to(artifact_dir) for path in artifact_dir.rglob("*") if path.is_file())
    files = [path for path in files if not any(part.startswith(".") for part in path.parts)]
    for path in files:
        (out_dir / path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(artifact_dir / path, out_dir / path)
    return [path.as_posix() for path in files]
