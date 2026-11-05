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
"""The label-free groupwise judge (Sec. 4.1; App. A.2, "Artifact comparison").

Candidate artifacts are compared in groups of four. Each candidate appears in
at least three groups, exposure differs by at most one, the co-comparison graph
stays connected when any one group or candidate is removed, and a search epoch
puts every child in a group with its parent. Each group is staged on its own:
the task as the worker saw it (task.md, deliverable_contract.json,
reference_files/) and the four artifacts under blinded labels artifact_A..D in a
seeded presentation order. The judge, the worker's model, inspects them with
local code execution and image viewing but no web, then submits ordered tiers,
hard-invalid labels and the decisive difference through finish_ranking (P6).
A group gets up to three attempts. Deblinded tiers are aggregated by
aggregate.aggregate.

    schedule(epoch_id, candidates, ...)   the balanced groups of a search epoch
    present(epoch_id, memberships, seed)  blinded labels and presentation for any groups
    rank(groups, artifacts, ...)          judge every group, then aggregate

Later-round learning reuses present and rank with its own groups, the unchanged
parent artifact (STOP) being one candidate. On Linux with bubblewrap, judge
shell commands run without network and with the task inputs mounted read-only;
elsewhere the judge only lacks a web tool.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import itertools
import json
import math
import os
import random
import re
import shutil
import signal
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import anyio
from pydantic import BaseModel, Field
from stirrup import Tool, ToolResult, ToolUseCountMetadata
from stirrup.tools.code_backends.base import CommandResult

from .. import planner, prompts
from ..domain import TaskPacket
from ..files import read_json, write_json
from ..llm import ModelProfile
from ..worker import WorkspaceCodeExec, run_agent, view_image
from . import aggregate

GROUP_SIZE, MIN_APPEARANCES = 4, 3
SEED = 1729                                         # default grouping and presentation seed
LABELS = ("artifact_A", "artifact_B", "artifact_C", "artifact_D")
SYSTEM_PROMPT = (
    "Judge the four blinded artifacts using only the uploaded task materials. "
    "Inspect every artifact, do not use the web, and never invoke a package manager, "
    "installer, or environment creator. Node dependencies are intentionally absent: "
    "inspect JavaScript and TypeScript statically without npm, npx, yarn, or pnpm. "
    "Before every shell call, check the entire chained command for prohibited steps. "
    "Finish only with finish_ranking."
)
CODE_EXEC_NOTE = (" Network and the host filesystem are unavailable. Never invoke pip, uv, conda, "
                  "mamba, apt, npm, yarn, pnpm, venv, virtualenv, or another installer. Hard limit: {uses} calls.")
INSTALL = re.compile(
    r"(?ix)"
    r"(?:\bpip3?\s+install\b|\bpython3?\s+-m\s+pip\s+install\b|"
    r"\buv\s+(?:pip|add)\b|\b(?:conda|mamba)\s+install\b|"
    r"\bapt(?:-get)?\b|\b(?:npm|npx|yarn|pnpm)\b|"
    r"\bvirtualenv\b|\bpython3?\s+-m\s+venv\b)"
)
BWRAP = shutil.which("bwrap")


@dataclass(frozen=True)
class Group:
    group_id: str
    candidates: tuple[str, ...]                     # membership order, the pair order of aggregation
    labels: dict[str, str]                          # candidate -> blinded label
    presentation: tuple[str, ...]                   # labels in the order the judge's manifest lists them


@dataclass(frozen=True)
class Material:
    """The task as the judge sees it: task.md, deliverable_contract.json, reference_files/."""
    text: str
    contract: dict
    references: tuple[Path, ...]


@dataclass(frozen=True)
class Judge:
    profile: ModelProfile                           # the worker's model, at the ranking temperature
    attempts: int = 3
    max_turns: int = 100
    code_exec_max_uses: int = 100
    turns_remaining_warning: int = 20
    context_summarization_cutoff: float = 0.9
    shell_timeout: int = 600
    timeout: float = 21600.0                        # seconds per attempt
    parallelism: int = 9                            # groups judged concurrently


def material(packet: TaskPacket, data_root: Path) -> Material:
    names = [Path(name).name for name in packet.deliverables]
    return Material(planner.task_query(packet, data_root),
                    {"expected_count": len(names), "expected_basenames": names,
                     "expected_suffixes": [Path(name).suffix.lower() for name in names]},
                    packet.references)


# ---- groups ------------------------------------------------------------------

def schedule(epoch_id: str, candidates: list[str], seed: int = SEED, required_pairs=(),
             parent_children: dict[str, list[str]] | None = None) -> list[Group]:
    """Groups for 5 to 12 candidates. Five candidates: one fixed candidate with each four-subset of the
    rest. Three parents with three children each: a fixed nine-group design. Otherwise a seeded greedy
    design that covers the required pairs first, retried until it is balanced and robust."""
    candidates = sorted(candidates)
    rng = random.Random(seed)
    if len(candidates) == 5:
        order = list(candidates)
        rng.shuffle(order)
        memberships = [tuple(candidate for candidate in order if candidate != omitted) for omitted in order[1:]]
    else:
        memberships = ((len(candidates) == 12 and _families(candidates, parent_children or {}))
                       or _balanced(candidates, {tuple(sorted(pair)) for pair in required_pairs}, rng))
    return present(epoch_id, memberships, seed)


def present(epoch_id: str, memberships: list[tuple[str, ...]], seed: int = SEED) -> list[Group]:
    groups = []
    for index, membership in enumerate(memberships, start=1):
        rng = random.Random(f"{seed}:{epoch_id}:{index}")
        order, labels = list(membership), list(LABELS)
        rng.shuffle(order)
        rng.shuffle(labels)
        groups.append(Group(f"{epoch_id}_group_{index:03d}", tuple(membership), dict(zip(order, labels)),
                            tuple(labels)))
    return groups


def _families(candidates: list[str], parent_children: dict[str, list[str]]) -> list[tuple[str, ...]] | None:
    if len(parent_children) != 3 or any(len(children) != 3 for children in parent_children.values()):
        return None
    parents = sorted(parent_children)
    (a1, a2, a3), (b1, b2, b3), (c1, c2, c3) = (sorted(parent_children[parent]) for parent in parents)
    if set(parents) | {a1, a2, a3, b1, b2, b3, c1, c2, c3} != set(candidates):
        return None
    p1, p2, p3 = parents
    return [(p1, a1, b1, c1), (p2, a1, b2, c2), (p2, a1, b3, c3), (p1, a2, b1, c2), (p3, a2, b2, c3),
            (p3, a2, b3, c1), (p2, a3, b1, c3), (p1, a3, b2, c1), (p3, a3, b3, c2)]


def _balanced(candidates: list[str], required: set[tuple[str, str]], rng: random.Random) -> list[tuple[str, ...]]:
    count = math.ceil(MIN_APPEARANCES * len(candidates) / GROUP_SIZE)
    every = list(itertools.combinations(candidates, GROUP_SIZE))
    for _ in range(2000):
        order = list(candidates)
        rng.shuffle(order)
        extra = set(order[:GROUP_SIZE * count - MIN_APPEARANCES * len(candidates)])
        target = {candidate: MIN_APPEARANCES + int(candidate in extra) for candidate in candidates}
        remaining, chosen, covered = dict(target), [], set()
        for _ in range(count):
            available = [group for group in every if group not in chosen and all(remaining[c] > 0 for c in group)]
            if not available:
                break
            rng.shuffle(available)

            def score(group):
                pairs = set(itertools.combinations(group, 2))
                return (len((required - covered) & pairs), len(pairs - covered), sum(remaining[c] for c in group),
                        -sum(target[c] - remaining[c] for c in group))

            group = max(available, key=score)
            chosen.append(group)
            covered |= set(itertools.combinations(group, 2))
            for candidate in group:
                remaining[candidate] -= 1
        if not any(remaining.values()) and required <= covered and robust(candidates, chosen):
            return chosen
    raise ValueError(f"no balanced group schedule for {len(candidates)} candidates")


def robust(candidates: list[str], groups: list[tuple[str, ...]]) -> bool:
    """Connected, also without any one group and without any one candidate."""
    def connected(rows, omitted=None):
        links = {candidate: set() for candidate in candidates}
        for group in rows:
            for left, right in itertools.combinations(group, 2):
                links[left].add(right)
                links[right].add(left)
        nodes = [candidate for candidate in candidates if candidate != omitted]
        seen, pending = set(), [nodes[0]]
        while pending:
            current = pending.pop()
            if current not in seen:
                seen.add(current)
                pending.extend(links[current] - {omitted})
        return seen == set(nodes)

    return (connected(groups) and all(connected(groups[:i] + groups[i + 1:]) for i in range(len(groups)))
            and all(connected(groups, candidate) for candidate in candidates))


# ---- staging -----------------------------------------------------------------

def instruction(group: Group) -> str:
    """P6 for one group."""
    return (prompts.load("learning/artifact_ranking.md").replace("{{OUTPUT_FILENAME}}", "group_ranking.json")
            .replace("{{GROUP_ID}}", group.group_id)
            .replace("{{ARTIFACT_LABELS}}", json.dumps(list(group.presentation), ensure_ascii=False)))


def stage(group: Group, artifacts: dict[str, Path], task: Material, visible: Path) -> None:
    """The judge-visible files of one group; inputs are read-only, scratch/ is the judge's."""
    for name in ("artifacts", "reference_files", "scratch"):
        (visible / name).mkdir(parents=True)
    (visible / "task.md").write_text(task.text.rstrip() + "\n", encoding="utf-8", newline="\n")
    write_json(visible / "deliverable_contract.json", task.contract)
    for source in task.references:
        _copy_read_only(source, visible / "reference_files" / source.name)
    for candidate in group.candidates:
        label = group.labels[candidate]
        _copy_read_only(artifacts[candidate], visible / "artifacts" / label / "files")
        files = _manifest(visible / "artifacts" / label / "files")
        digest = hashlib.sha256(json.dumps(files, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
        write_json(visible / "artifacts" / label / "manifest.json", {"label": label, "files": files,
                                                                     "manifest_digest": digest})
        (visible / "artifacts" / label / "manifest.json").chmod(0o444)
    write_json(visible / "group_manifest.json", {"group_id": group.group_id, "artifacts": [
        {"label": label, "path": f"artifacts/{label}"} for label in group.presentation]})
    (visible / "INSTRUCTION.md").write_text(instruction(group), encoding="utf-8", newline="\n")
    for name in ("task.md", "deliverable_contract.json", "group_manifest.json", "INSTRUCTION.md"):
        (visible / name).chmod(0o444)


def _copy_read_only(source: Path, target: Path) -> None:
    if source.is_dir():
        shutil.copytree(source, target)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    for path in [target, *target.rglob("*")] if target.is_dir() else [target]:
        path.chmod(0o555 if path.is_dir() else 0o444)


def _manifest(root: Path) -> list[dict]:
    return [{"path": path.relative_to(root).as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
             "bytes": path.stat().st_size} for path in sorted(root.rglob("*")) if path.is_file()]


# ---- the judge session -------------------------------------------------------

class RankingFinishParams(BaseModel):
    # No docstring: it would enter the tool schema the judge sees.
    group_id: Annotated[str, Field(min_length=1, description="The exact visible group ID.")]
    ranking_tiers: Annotated[
        list[list[str]],
        Field(min_length=1, description="Artifact-label tiers ordered from best to worst."),
    ]
    hard_invalid: Annotated[list[str], Field(description="Hard-invalid labels, possibly empty.")]
    decisive_difference: Annotated[
        str,
        Field(min_length=1, description="Concise task-grounded reason for the ordering."),
    ]


def ranking_error(payload: RankingFinishParams, group_id: str) -> str | None:
    """The finish-tool message for an invalid ranking (model-facing text), else None."""
    labels = list(LABELS)
    if payload.group_id != group_id:
        return f"group_id must be exactly {group_id!r}."
    if any(not tier for tier in payload.ranking_tiers):
        return "ranking_tiers must not contain an empty tier."
    flattened = [label for tier in payload.ranking_tiers for label in tier]
    if len(flattened) != len(set(flattened)):
        return "Every artifact label must appear exactly once; duplicate labels were found."
    if set(flattened) != set(labels):
        return f"ranking_tiers must partition exactly these labels: {labels}."
    if len(payload.hard_invalid) != len(set(payload.hard_invalid)):
        return "hard_invalid must not contain duplicates."
    if not set(payload.hard_invalid).issubset(labels):
        return "hard_invalid may contain only visible artifact labels."
    if not payload.decisive_difference.strip():
        return "decisive_difference must be nonempty."
    return None


def finish_tool(group_id: str) -> Tool[RankingFinishParams, ToolUseCountMetadata]:
    async def executor(params: RankingFinishParams) -> ToolResult[ToolUseCountMetadata]:
        error = ranking_error(params, group_id)
        return ToolResult(content=error or "ranking accepted", success=error is None, metadata=ToolUseCountMetadata())

    return Tool[RankingFinishParams, ToolUseCountMetadata](
        name="finish_ranking",
        description=("Submit the complete groupwise ranking after inspecting every artifact. "
                     "Every visible label must occur exactly once in the ordered tiers."),
        parameters=RankingFinishParams,
        executor=executor,
    )


class JudgeCodeExec(WorkspaceCodeExec):
    """code_exec for the judge: no package installs, a budget of calls, and under bubblewrap no
    network, with artifacts/ and reference_files/ mounted read-only at /workspace."""

    def __init__(self, workspace: Path, shell_timeout: int, max_uses: int) -> None:
        super().__init__(workspace, shell_timeout=shell_timeout)
        self.max_uses, self.uses = max_uses, 0

    def get_code_exec_tool(self, *, name: str = "code_exec", description: str | None = None) -> Tool:
        base = super().get_code_exec_tool(name=name, description=description)

        async def executor(params):
            self.uses += 1
            if self.uses > self.max_uses:
                return ToolResult(content=f"Inspection budget exhausted after {self.max_uses} code_exec calls; "
                                          "finish now.", success=False, metadata=ToolUseCountMetadata(num_uses=1))
            return await base.executor(params)

        return Tool(name=base.name, description=base.description + CODE_EXEC_NOTE.format(uses=self.max_uses),
                    parameters=base.parameters, executor=executor)

    def _argv(self, cmd: str) -> list[str]:
        if not BWRAP:
            return ["bash", "-c", cmd]
        prefix = str(Path(sys.prefix).resolve())
        argv = ["bwrap", "--unshare-net", "--die-with-parent", "--new-session",
                "--ro-bind", "/usr", "/usr", "--ro-bind", "/bin", "/bin", "--ro-bind", "/lib", "/lib",
                "--ro-bind-try", "/lib64", "/lib64", "--ro-bind", "/etc", "/etc", "--dev", "/dev", "--proc", "/proc",
                "--tmpfs", "/tmp", "--tmpfs", "/run"]
        for root in sorted({prefix, str(Path(sys.base_prefix).resolve())}):     # a venv links to its base
            argv += ["--ro-bind", root, root]
        argv += ["--bind", str(self._temp_dir), "/workspace"]
        for name in ("artifacts", "reference_files"):
            if (self._temp_dir / name).exists():
                argv += ["--ro-bind", str(self._temp_dir / name), f"/workspace/{name}"]
        return argv + ["--chdir", "/workspace", "--setenv", "HOME", "/workspace",
                       "--setenv", "PATH", f"{prefix}/bin:/usr/bin:/bin", "--setenv", "LANG", "C.UTF-8",
                       "--setenv", "PYTHONDONTWRITEBYTECODE", "1",
                       "--setenv", "XDG_CACHE_HOME", "/workspace/scratch/.cache", "bash", "-c", cmd]

    async def run_command(self, cmd: str, *, timeout: int | None = None) -> CommandResult:
        timeout = self._shell_timeout if timeout is None else timeout
        if INSTALL.search(cmd):
            return CommandResult(exit_code=1, stdout="", stderr="Package installation is disabled for this judgment.",
                                 error_kind="package_install_disabled")
        env = ({"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"} if BWRAP
               else {key: value for key, value in os.environ.items() if key != "GEMINI_API_KEY"})
        process = None
        try:
            with anyio.fail_after(timeout):
                process = await anyio.open_process(self._argv(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                                   cwd=self._temp_dir, env=env, start_new_session=True)
                stdout: list[bytes] = []
                stderr: list[bytes] = []

                async def drain(stream, chunks: list[bytes]) -> None:
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
            return CommandResult(exit_code=1, stdout="", stderr=f"Command timed out after {timeout} seconds",
                                 error_kind="timeout")
        except Exception as exc:
            return CommandResult(exit_code=1, stdout="", stderr=str(exc), error_kind="execution_error")


def _session(group: Group, visible: Path, attempt_dir: Path, judge: Judge) -> dict | None:
    """One judge session. Stirrup copies the judge-visible files into a fresh workspace and lists
    them in its system prompt. Returns the finish_ranking arguments."""
    code_exec = JudgeCodeExec(attempt_dir / "workspace", judge.shell_timeout, judge.code_exec_max_uses)
    return run_agent(judge.profile, system_prompt=SYSTEM_PROMPT, prompt=instruction(group),
                     tools=[code_exec, view_image()] if judge.profile.images else [code_exec],
                     finish_tools=finish_tool(group.group_id), logs=attempt_dir / "logs",
                     name="independent_groupwise_artifact_judge", max_turns=judge.max_turns,
                     turns_remaining_warning_threshold=judge.turns_remaining_warning,
                     context_summarization_cutoff=judge.context_summarization_cutoff,
                     input_files=sorted(visible.iterdir()), timeout=judge.timeout).finish


def judge_group(group: Group, group_dir: Path, judge: Judge) -> dict:
    """The validated finish_ranking arguments of one staged group, from the first of up to
    `judge.attempts` sessions that returns a valid ranking. Raises when every attempt fails."""
    errors = []
    for attempt in range(1, judge.attempts + 1):
        try:
            finish = _session(group, group_dir / "judge_visible", group_dir / f"attempt_{attempt:02d}", judge)
            payload = RankingFinishParams.model_validate(finish)
            error = ranking_error(payload, group.group_id)
        except Exception as exc:                    # a failed session or a missing or malformed finish
            error = repr(exc)
        if error is None:
            return payload.model_dump(mode="json")
        errors.append(f"attempt {attempt}: {error}")
    raise RuntimeError(f"{group.group_id}: no valid ranking: {errors}")


def rank(groups: list[Group], artifacts: dict[str, Path], *, task: Material, judge: Judge, out_dir: Path) -> dict:
    """Judge every group (each result is kept in out_dir/<group_id>/ranking.json and reused) and
    aggregate the deblinded tiers: aggregate.aggregate's fields plus the judged "groups"."""
    def one(group: Group) -> dict:
        group_dir = out_dir / group.group_id
        if not (group_dir / "ranking.json").exists():
            if not (group_dir / "judge_visible" / "INSTRUCTION.md").exists():
                stage(group, artifacts, task, group_dir / "judge_visible")
            write_json(group_dir / "ranking.json", judge_group(group, group_dir, judge))
        payload = read_json(group_dir / "ranking.json")
        candidate = {label: name for name, label in group.labels.items()}
        return {"group_id": group.group_id, "candidates": list(group.candidates), "labels": group.labels,
                "presentation": list(group.presentation),
                "tiers": [[candidate[label] for label in tier] for tier in payload["ranking_tiers"]],
                "hard_invalid": [candidate[label] for label in payload["hard_invalid"]],
                "decisive_difference": payload["decisive_difference"].strip()}

    with ThreadPoolExecutor(max_workers=max(1, min(judge.parallelism, len(groups)))) as pool:
        rows = list(pool.map(one, groups))
    return {"groups": rows, **aggregate.aggregate(rows)}
