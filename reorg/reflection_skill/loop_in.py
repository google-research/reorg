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
"""Loop-In (Sec. 3.3, App. A.1): a stakeholder consults another stakeholder.

    select     the initiator picks a partner (any other catalog role) and opens
    partner    the partner takes one discussion turn
    initiator  the initiator takes one discussion turn
    publish    the initiator publishes the synthesis ("# Handoff Summary")

Each phase is one model session run by the domain's driver, which owns
sandboxes, tools and the phase prompt wrapper; this module owns the protocol,
the prompt variables and the published transcript.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol

from .. import prompts
from .graph import ROLES

PAIRED_ROUNDS = 1                                  # one partner turn, one initiator turn
_VARIABLE = re.compile(r"{{\s*([A-Za-z_][A-Za-z0-9_]*)\s*}}")


@dataclass(frozen=True)
class Message:
    index: int
    round_index: int                               # 0 is the opening message
    participant: str                               # "initiator" | "partner"
    display_name: str
    text: str


@dataclass
class LoopInResult:
    status: str                                    # "ok" | "partial"
    phase: str
    initiator: str
    partner: str | None = None
    messages: list[Message] = field(default_factory=list)
    summary: str | None = None
    error: str | None = None


class Driver(Protocol):
    """Runs one phase session for a participant (role ID) with a rendered protocol prompt."""

    def select(self, initiator: str, protocol_prompt: str) -> tuple[str, str]: ...      # (partner display name, opening)
    def turn(self, speaker: str, protocol_prompt: str, label: str) -> str: ...          # message text
    def publish(self, initiator: str, protocol_prompt: str) -> str: ...                 # handoff markdown


def render(name: str, values: dict) -> str:
    """Fill a {{ variable }} protocol template; every declared variable must be given, and only those."""
    source = prompts.load(name).strip()
    declared = set(_VARIABLE.findall(source))
    if declared != set(values):
        raise ValueError(f"{name}: variables {sorted(declared)} != given {sorted(values)}")
    return _VARIABLE.sub(lambda match: str(values[match.group(1)]), source)


def transcript(messages: list[Message]) -> str:
    return "\n\n".join(f"Message {m.index} - {m.display_name}:\n{m.text}" for m in messages) or "No messages yet."


def partners(initiator: str) -> list[str]:
    """Eligible partner display names: every other catalog role, in catalog order."""
    return [name for role, name in ROLES.items() if role != initiator]


def run(initiator: str, driver: Driver) -> LoopInResult:
    result = LoopInResult(status="partial", phase="select", initiator=initiator)
    by_name = {ROLES[role]: role for role in ROLES if role != initiator}
    try:
        name, opening = driver.select(initiator, render("loop_in/select.md", {
            "eligible_partner_display_names": "\n".join(f"- {name}" for name in partners(initiator))}))
        name, opening = name.strip(), opening.strip()
        if name not in by_name or not opening:
            raise ValueError(f"partner must be one of {sorted(by_name)} with a nonempty opening")
        result.partner = by_name[name]
        result.messages.append(Message(1, 0, "initiator", ROLES[initiator], opening))
        for round_index in range(1, PAIRED_ROUNDS + 1):
            for participant, speaker, other in (("partner", result.partner, initiator),
                                                ("initiator", initiator, result.partner)):
                result.phase = f"round_{round_index}_{participant}"
                text = driver.turn(speaker, render("loop_in/discussion.md", {
                    "participant_display_name": ROLES[speaker],
                    "other_participant_display_name": ROLES[other],
                    "round_index": round_index,
                    "configured_rounds": PAIRED_ROUNDS,
                    "transcript": transcript(result.messages)}), f"{participant}_{round_index}").strip()
                if not text:
                    raise ValueError(f"{participant} message is empty")
                result.messages.append(Message(len(result.messages) + 1, round_index, participant, ROLES[speaker], text))
        result.phase = "publish"
        summary = driver.publish(initiator, render("loop_in/summary.md", {"transcript": transcript(result.messages)})).strip()
        if not summary:
            raise ValueError("published handoff is empty")
        result.summary, result.status, result.phase = summary, "ok", "complete"
    except Exception as exc:                       # a failed phase leaves a partial, typed result
        result.error = repr(exc)
    return result


def outcome(result: LoopInResult) -> str:
    """The published contribution: the initiator's synthesis followed by the discussion transcript."""
    blocks = [(result.summary or "# Handoff Summary\n\nmissing").rstrip(), "", "# Discussion Transcript"]
    for message in result.messages:
        stage = "Opening" if message.round_index == 0 else f"Round {message.round_index}"
        blocks += ["", f"## {stage} - {message.display_name}", "", message.text]
    return "\n".join(blocks).strip() + "\n"
