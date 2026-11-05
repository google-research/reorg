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
"""Panel decision and GDPval score (App. A.5.2).

Each judge's points are mapped back from the blinded A/B labels to candidate
vs expert. The panel outcome is the category (win, tie, loss) with a strict
majority of the three votes; a win/tie/loss split is a tie. A comparison with
any missing or failed judgment has no outcome. Raw points are never averaged
across judges. Over a cohort of N_cohort tasks,

    WinTie = (N_win + N_tie) / N_cohort,

so tasks without a submission or an outcome count against the rate.
"""

from __future__ import annotations

from collections import Counter

OUTCOMES = ("win", "tie", "loss")


def candidate_outcome(row: dict) -> str:
    """win / tie / loss of the candidate against the expert in one judgment."""
    points = {row["mapping"][label]["kind"]: row[f"response_{label.lower()}_raw_points"] for label in "AB"}
    if points["submission"] > points["expert"]:
        return "win"
    return "loss" if points["submission"] < points["expert"] else "tie"


def majority(results: dict[str, list[dict]]) -> dict[str, dict]:
    """results: member -> judgment rows. Returns task_id -> panel decision."""
    by_member = {member: {row["task_id"]: row for row in rows} for member, rows in results.items()}
    decisions = {}
    for task_id in dict.fromkeys(row["task_id"] for rows in results.values() for row in rows):
        rows = {member: index.get(task_id) for member, index in by_member.items()}
        failed = [member for member, row in rows.items() if row is None or row["status"] != "ok"]
        if failed:
            decisions[task_id] = {"outcome": None, "reason": "incomplete_panel:" + ",".join(failed)}
            continue
        votes = {member: candidate_outcome(row) for member, row in rows.items()}
        counts = Counter(votes.values())
        winner = next((key for key in OUTCOMES if counts[key] > len(votes) // 2), None)
        decisions[task_id] = {"outcome": winner or "tie", "votes": votes,
                              "reason": None if winner else "no_strict_majority"}
    return decisions


def win_tie(decisions: dict[str, dict], cohort: list[str]) -> dict:
    counts = Counter((decisions.get(task_id) or {}).get("outcome") or "unavailable" for task_id in cohort)
    return {"cohort": len(cohort), **{key: counts[key] for key in (*OUTCOMES, "unavailable")},
            "win_tie_rate": (counts["win"] + counts["tie"]) / len(cohort)}
