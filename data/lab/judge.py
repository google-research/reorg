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
"""LAB evaluation (App. A.5.3): harvey-labs' criterion judge with Gemini 3.1 Pro.

Upstream's `evaluation.scoring.score_rubric` (the pinned checkout under
<data>/lab/harvey-labs) loads the deliverables each criterion names and asks
the judge for a pass/fail verdict and rationale per criterion, using its
`rubric_criterion` template with the criterion's matching instructions and the
task description. Upstream supplies only that template; this wrapper adds a
one-line system prompt and calls Gemini 3.1 Pro
(reorg.llm.gemini_generate) at temperature 0 with high thinking, 16,384
output tokens and a JSON verdict schema. Per task t with p_t of n_t criteria
passed,

    S_macro = (1/T) sum_t p_t / n_t,        S_pooled = sum_t p_t / sum_t n_t,

over the T eligible tasks, plus the number of tasks passing every criterion.
An incomplete submission gets p_t = 0 and is ineligible. A missing run record
or a failed judgment (including an extraction failure) is unavailable, never a
model zero: the extraction guard turns upstream's silent read errors into
failures, while upstream's own DOCX-parse rejection stays a model failure.
DOCX extraction needs pandoc on PATH.

    reads   <run>/<task_id>/result.json ("status"), <run>/<task_id>/final/submission/
    writes  <run>/eval/<task_id>/{scores.json, model_calls.jsonl}, <run>/eval/results.jsonl, <run>/eval/summary.json
"""

from __future__ import annotations

import importlib
import json
import re
import shutil
import statistics
import sys
from pathlib import Path

from reorg import llm
from reorg.domain import ROOT
from reorg.files import read_json, write_json, write_jsonl

JUDGE_SYSTEM_PROMPT = """You are grading one Harvey LAB rubric criterion. Apply only the supplied criterion to the supplied artifact text. Return exactly one JSON object with keys `verdict` (`pass` or `fail`) and `reasoning` (a concise explanation). Do not add markdown."""
VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["pass", "fail"]},
        "reasoning": {"type": "string"},
    },
    "required": ["verdict", "reasoning"],
}


class ArtifactExtractionError(RuntimeError):
    pass


def native_docx_parse_error(path: Path, text: str) -> bool:
    """Upstream's own rejection of a DOCX it cannot parse (a model failure)."""
    return (Path(path).suffix.lower() == ".docx" and text.startswith("(error reading ")
            and ": pandoc failed: couldn't parse docx file" in text)


def guard(scoring) -> None:
    """Make upstream's silent read errors raise instead of being judged as text."""
    read = scoring._read_file_as_text

    def checked_read(path, **kwargs):
        text = read(path, **kwargs)
        if text.startswith("(error reading ") and not native_docx_parse_error(path, text):
            raise ArtifactExtractionError(f"Artifact extraction failed for {path}: {text}")
        return text

    scoring._read_file_as_text = checked_read
    # Upstream's last-resort file matcher calls Claude through the Anthropic SDK and finds no
    # match without Anthropic credentials. It is disabled, so LAB judging stays on Gemini alone.
    scoring._llm_match_deliverables = lambda *args, **kwargs: {}


def parse_verdict(text: str) -> dict:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None
    if not isinstance(payload, dict):
        # A reply cut off after its verdict still counts.
        match = re.search(r'"verdict"\s*:\s*"(pass|fail)"', text, flags=re.IGNORECASE)
        if match is None:
            raise ValueError(f"Judge returned no JSON object: {text[:300]}")
        payload = {"verdict": match.group(1), "reasoning": "Judge response was truncated after emitting its verdict."}
    verdict = str(payload.get("verdict") or "").lower()
    if verdict not in {"pass", "fail"}:
        raise ValueError(f"Invalid verdict: {verdict!r}")
    return {"verdict": verdict, "reasoning": str(payload.get("reasoning") or "")}


class GeminiJudge:
    """The judge object upstream's score_rubric calls: evaluate_from_file(prompt_name, variables)."""

    def __init__(self, profile: dict, prompts_dir: Path, log_path: Path) -> None:
        self.profile = profile
        self.prompts_dir = prompts_dir
        self.log_path = log_path

    def payload(self, prompt: str) -> dict:
        return {
            "systemInstruction": {"parts": [{"text": JUDGE_SYSTEM_PROMPT}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "maxOutputTokens": self.profile["max_tokens"],
                "temperature": self.profile["temperature"],
                "thinkingConfig": {"thinkingLevel": self.profile["thinking_level"]},
                "responseMimeType": "application/json",
                "responseSchema": VERDICT_SCHEMA,
            },
        }

    def evaluate_from_file(self, prompt_name: str, variables: dict) -> dict:
        prompt = (self.prompts_dir / f"{prompt_name}.txt").read_text(encoding="utf-8").format(**variables)
        for attempt in range(self.profile["attempts"]):
            try:
                data = llm.gemini_generate(self.profile["model"], self.payload(prompt),
                                           timeout=self.profile["request_timeout"], log_path=self.log_path)
                parts = ((data.get("candidates") or [{}])[0].get("content") or {}).get("parts") or []
                return parse_verdict("".join(str(part.get("text") or "") for part in parts if not part.get("thought")))
            except Exception as exc:
                error = exc
        raise RuntimeError(f"Judge failed after {self.profile['attempts']} attempts: {error!r}")


def score_task(scoring, profile: dict, harvey: Path, task_id: str, run_dir: Path, out: Path) -> dict:
    config = read_json(harvey / "tasks" / task_id / "task.json")
    record = run_dir / task_id / "result.json"
    row = {"task_id": task_id, "n_criteria": len(config["criteria"])}
    if not record.is_file():
        return {**row, "status": "unavailable", "eligible": False}
    if (status := read_json(record)["status"]) != "complete":
        failed = [{"id": criterion["id"], "title": criterion["title"], "verdict": "fail",
                   "reasoning": f"Solver submission was not complete: {status}."} for criterion in config["criteria"]]
        return {**row, "status": "incomplete", "eligible": False, "n_passed": 0, "all_pass": False,
                "criteria_results": failed}
    # Upstream reads <stage>/output/.
    stage = out / task_id / "upstream_run"
    shutil.rmtree(stage, ignore_errors=True)
    (stage / "output").mkdir(parents=True)
    submission = run_dir / task_id / "final" / "submission"
    if submission.is_dir():
        shutil.copytree(submission, stage / "output", dirs_exist_ok=True)
    judge = GeminiJudge(profile, harvey / "evaluation" / "prompts", out / task_id / "model_calls.jsonl")
    try:
        result = scoring.score_rubric(criteria=config["criteria"], run_dir=stage, judge=judge,
                                      task_desc=str(config["title"]), parallel=profile["criterion_parallelism"])
    except Exception as exc:
        return {**row, "status": "unavailable", "eligible": False, "error": repr(exc)}
    n_passed = sum(criterion["verdict"] == "pass" for criterion in result.criteria_results)
    return {**row, "status": "ok", "eligible": True, "n_passed": n_passed,
            "all_pass": result.score == result.max_score, "criteria_results": result.criteria_results}


def aggregate(rows: list[dict]) -> dict:
    eligible = [row for row in rows if row["eligible"]]
    passed = sum(row["n_passed"] for row in eligible)
    total = sum(row["n_criteria"] for row in eligible)
    return {
        "tasks": len(rows),
        "eligible": len(eligible),
        "incomplete": sum(row["status"] == "incomplete" for row in rows),
        "unavailable": sum(row["status"] == "unavailable" for row in rows),
        "macro": statistics.fmean(row["n_passed"] / row["n_criteria"] for row in eligible) if eligible else None,
        "pooled": passed / total if total else None,
        "criteria_passed": passed,
        "criteria": total,
        "all_pass": sum(row["all_pass"] for row in eligible),
    }


def evaluate(run_dir: Path, task_ids: list[str], data: Path) -> dict:
    """data: the prepared <data>/lab with its harvey-labs checkout."""
    if shutil.which("pandoc") is None:
        raise SystemExit("LAB judging needs pandoc on PATH (upstream extracts DOCX with it)")
    harvey = data / "harvey-labs"
    sys.path.insert(0, str(harvey))
    scoring = importlib.import_module("evaluation.scoring")
    guard(scoring)
    profile = read_json(ROOT / "configs" / "judges" / "lab_gemini31pro.json")
    out = run_dir / "eval"
    rows = []
    for task_id in task_ids:
        row = score_task(scoring, profile, harvey, task_id, run_dir, out)
        write_json(out / task_id / "scores.json", row)
        rows.append(row)
        print(f"{task_id}: {row['status']} {row.get('n_passed')}/{row['n_criteria']}", flush=True)
    write_jsonl(out / "results.jsonl", rows)
    summary = aggregate(rows)
    write_json(out / "summary.json", summary)
    return summary
