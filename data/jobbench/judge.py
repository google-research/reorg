#!/usr/bin/env python3
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
"""JobBench evaluation (App. A.5.3): the pinned upstream rubric judge with a
Gemini 3.5 Flash transport.

Upstream (Job-Bench/job-bench-eval @ 007b8064, eval/judge.py) extracts the
text of the submitted files, asks the judge for one JSON verdict per rubric
item (with images when an item needs visual inspection), and scores a task as
earned weight / total weight; an item earns its weight only when all its
criteria pass. Its prompts and scoring are used as is. On top of upstream,
this wrapper adds:

  * a Gemini transport behind upstream's OpenAI client: high thinking, a JSON
    response schema, and JSON_FORMAT_GUARD appended to upstream's system prompt;
  * a 150,000-character cap on the extracted contents (head and tail per file);
  * status rules: a non-complete, empty or unreadable submission is a model
    zero; an infrastructure failure (solver infra_error, a missing converter, a
    failed judge call) leaves the task unavailable.

The score is the macro mean of normalized task scores over eligible tasks,
overall and per subset; with all 128 tasks eligible it equals
S_JobBench = (65 S_main + 63 S_easy) / 128.

The judge runs in upstream's own locked environment (`uv sync` plus xlrd), so
this file uses only the standard library and upstream. `evaluate` fetches and
prepares that environment and re-runs this file inside it:

    reads   <run>/<task_id>/result.json ("status"), <run>/<task_id>/final/submission/
    writes  <run>/eval/<task_id>/{rubrics.json, calls.jsonl}, <run>/eval/results.jsonl, <run>/eval/summary.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
SETTINGS = ROOT / "configs" / "judges" / "jobbench_gemini35flash.json"
UPSTREAM = "https://github.com/Job-Bench/job-bench-eval.git"
COMMIT = "007b80648929131bd6aa8d01321360217f0f379e"
# Upstream reads .xls (Excel 97-2003) files through pandas, but its lockfile omits the engine.
EXTRA = "xlrd==2.0.2"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com"

JSON_FORMAT_GUARD = (
    "Keep every reasoning and evidence string concise and use plain text for mathematical symbols. "
    "Never repeat a character or Unicode escape sequence more than eight times."
)
MAX_STRUCTURED_STRING_LENGTH = 4096
MISSING_CONVERTER = re.compile(r"\[ERROR:[^\]\n]*\bnot available (?:for|to read)\b", re.IGNORECASE)
RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


# ---- core side: fetch upstream, build its environment, run this file there --

def checkout(data: Path) -> Path:
    """<data>/jobbench/job-bench-eval at the pinned commit (rerunnable)."""
    root = data / "job-bench-eval"
    if not (root / ".git").is_dir():
        subprocess.run(["git", "clone", "--no-checkout", UPSTREAM, str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "checkout", "--force", "--detach", COMMIT], check=True)
    return root


def environment(root: Path) -> Path:
    """Upstream's uv.lock environment plus xlrd, created once; returns its python."""
    python = root / ".venv" / "bin" / "python"
    if not python.exists():
        subprocess.run(["uv", "sync", "--frozen", "--python", "3.13"], cwd=root, check=True)
        subprocess.run(["uv", "pip", "install", "--python", str(python), EXTRA], cwd=root, check=True)
    return python


def evaluate(run_dir: Path, task_ids: list[str], data: Path) -> dict:
    """data: the prepared <data>/jobbench. Returns the summary (aggregate)."""
    python = environment(checkout(data))
    subprocess.run([str(python), str(Path(__file__).resolve()), "--run", str(run_dir), "--data", str(data),
                    *task_ids], check=True)
    return json.loads((run_dir / "eval" / "summary.json").read_text(encoding="utf-8"))


# ---- upstream side ---------------------------------------------------------

def cap_contents(contents: str, max_chars: int) -> str:
    """Fit the extracted contents into max_chars. Files that fit an equal share
    stay whole and pass their unused share on; every other file keeps its head
    (3/5) and tail around a truncation marker."""
    if len(contents) <= max_chars:
        return contents
    sections = [section for section in re.split(r"(?=^=== FILE: )", contents, flags=re.MULTILINE) if section]
    allocations = [0] * len(sections)
    active = set(range(len(sections)))
    remaining = max_chars
    while active:
        share = remaining // len(active)
        small = [index for index in active if len(sections[index]) <= share]
        if not small:
            for index in sorted(active):
                allocations[index] = share
            for index in sorted(active)[: remaining - share * len(active)]:
                allocations[index] += 1
            break
        for index in small:
            allocations[index] = len(sections[index])
            remaining -= allocations[index]
            active.remove(index)
    capped = []
    for section, allocation in zip(sections, allocations, strict=True):
        if len(section) <= allocation:
            capped.append(section)
            continue
        marker = "\n... [Gemini evidence truncated for context; middle omitted] ...\n"
        budget = allocation - len(marker)
        head = max(0, budget * 3 // 5)
        tail = max(0, budget - head)
        capped.append(section[:head] + marker + (section[-tail:] if tail else ""))
    return "".join(capped)


def _text(content) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(str(part.get("text") or "") for part in content if part.get("type") == "text")


def _user_parts(content) -> list[dict]:
    if isinstance(content, str):
        return [{"text": content}]
    parts = []
    for block in content:
        if block["type"] == "text":
            parts.append({"text": str(block.get("text") or "")})
        else:  # image_url: upstream only sends base64 data URLs
            mime, data = re.fullmatch(r"data:([^;,]+);base64,(.+)", block["image_url"]["url"], flags=re.DOTALL).groups()
            parts.append({"inlineData": {"mimeType": mime, "data": data}})
    return parts


def _response_schema(prompt: str) -> dict:
    count = int(re.search(r"criteria_results array must have exactly (\d+) items", prompt).group(1))
    text = {"type": "STRING", "maxLength": MAX_STRUCTURED_STRING_LENGTH}
    return {
        "type": "OBJECT",
        "properties": {
            "criteria_results": {
                "type": "ARRAY",
                "minItems": count,
                "maxItems": count,
                "items": {
                    "type": "OBJECT",
                    "properties": {"index": {"type": "INTEGER"}, "passed": {"type": "BOOLEAN"},
                                   "reasoning": text, "evidence": text},
                    "required": ["index", "passed", "reasoning", "evidence"],
                },
            },
            "rubric_passed": {"type": "BOOLEAN"},
            "overall_reasoning": text,
        },
        "required": ["criteria_results", "rubric_passed", "overall_reasoning"],
    }


def _close_truncated_json(text: str) -> tuple[str, bool]:
    """Close the open containers of a reply cut off at MAX_TOKENS, if that yields
    a complete verdict object; never invents truncated string content."""
    try:
        json.loads(text)
        return text, False
    except json.JSONDecodeError:
        pass
    stack: list[str] = []
    in_string = escaped = False
    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char in "[{":
            stack.append(char)
        elif char in "]}":
            if not stack or stack.pop() != {"}": "{", "]": "["}[char]:
                return text, False
    if in_string or not stack:
        return text, False
    repaired = text + "".join("}" if opener == "{" else "]" for opener in reversed(stack))
    try:
        payload = json.loads(repaired)
    except json.JSONDecodeError:
        return text, False
    if not isinstance(payload, dict) or not {"criteria_results", "rubric_passed", "overall_reasoning"} <= set(payload):
        return text, False
    return repaired, True


class GeminiTransport:
    """Upstream's `client.chat.completions.create(...)` answered by one Gemini
    generateContent request; it stands in for `upstream.get_openai_client`."""

    def __init__(self, settings: dict) -> None:
        self.model = settings["model"]
        self.thinking_level = settings["thinking_level"]
        self.max_tokens = settings["max_tokens"]
        self.api_key = os.environ["GEMINI_API_KEY"]
        self.log_path: Path | None = None   # set per task; tasks run one at a time
        self._lock = threading.Lock()

    def client(self, api_base: str, api_key: str) -> SimpleNamespace:
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=self.create)))

    def payload(self, messages: list[dict], temperature: float) -> dict:
        system = "\n\n".join(_text(message["content"]) for message in messages if message["role"] == "system")
        system = f"{system}\n\n{JSON_FORMAT_GUARD}" if system else JSON_FORMAT_GUARD
        (user,) = [message["content"] for message in messages if message["role"] == "user"]
        return {
            "contents": [{"role": "user", "parts": _user_parts(user)}],
            "generationConfig": {
                "maxOutputTokens": self.max_tokens,
                "temperature": float(temperature),
                "thinkingConfig": {"thinkingLevel": self.thinking_level},
                "responseMimeType": "application/json",
                "responseSchema": _response_schema(_text(user)),
            },
            "systemInstruction": {"parts": [{"text": system}]},
        }

    def _log(self, row: dict) -> None:
        with self._lock:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"model": self.model, **row}, ensure_ascii=False) + "\n")

    def _post(self, body: bytes, timeout: float) -> dict:
        request = urllib.request.Request(
            f"{GEMINI_BASE_URL}/v1beta/models/{self.model}:generateContent", data=body, method="POST",
            headers={"Content-Type": "application/json", "Accept-Encoding": "identity", "x-goog-api-key": self.api_key})
        for attempt in range(1, 6):
            started = time.time()
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                error = exc.read().decode("utf-8", errors="replace")
                self._log({"status_code": exc.code, "latency_seconds": round(time.time() - started, 3),
                           "error_body": error[:4000]})
                if exc.code not in RETRYABLE_STATUS or attempt == 5:
                    raise RuntimeError(f"Gemini HTTP {exc.code}: {error[:1000]}") from exc
                delay = min(120, 15 * 2 ** (attempt - 1)) if exc.code == 429 else min(10, 2 ** (attempt - 1))
            except (urllib.error.URLError, TimeoutError) as exc:
                self._log({"error": repr(exc)})
                if attempt == 5:
                    raise
                delay = min(10, 2 ** (attempt - 1))
            time.sleep(delay)

    def create(self, *, messages: list[dict], temperature: float, timeout: float, **_) -> SimpleNamespace:
        started = time.time()
        data = self._post(json.dumps(self.payload(messages, temperature)).encode("utf-8"), timeout)
        candidate = (data.get("candidates") or [{}])[0]
        parts = (candidate.get("content") or {}).get("parts") or []
        text = "".join(str(part.get("text") or "") for part in parts if not part.get("thought")).strip()
        if not text:
            raise RuntimeError(f"Gemini returned no answer text: {data}")
        repaired = False
        if candidate.get("finishReason") == "MAX_TOKENS":
            text, repaired = _close_truncated_json(text)
        self._log({"status_code": 200, "latency_seconds": round(time.time() - started, 3),
                   "usage": data.get("usageMetadata") or {}, "finish_reason": candidate.get("finishReason"),
                   "response_repaired": repaired, "response_text": text, "raw_response": data})
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


def judge_task(upstream: ModuleType, rubrics: list[dict], submission: Path, submission_status: str,
               settings: dict) -> tuple[str, list[dict]]:
    """(status, upstream-shaped rubric results); status is ok, model_failure or infra_error."""
    def failed(reason: str) -> list[dict]:
        return [upstream.build_failed_rubric_result(index, rubric, reason) for index, rubric in enumerate(rubrics)]

    files = sorted(path for path in submission.rglob("*") if path.is_file()) if submission.is_dir() else []
    if submission_status == "infra_error":
        return "infra_error", failed("Solver submission failed because of infrastructure; no model score is assigned.")
    if submission_status != "complete":
        return "model_failure", failed(
            f"Solver submission status was {submission_status!r}; partial outputs were not judged.")
    if not files:
        return "model_failure", failed("No output files found in the model output directory.")
    if all(path.stat().st_size == 0 for path in files):
        return "model_failure", failed("All output files were empty.")
    contents = cap_contents(upstream.extract_all_file_contents(submission), settings["max_file_content_chars"])
    if MISSING_CONVERTER.search(contents):
        return "infra_error", failed("Required artifact converter is unavailable.")
    if not contents.strip():
        return "model_failure", failed("Output files were unreadable or empty after conversion.")
    images = upstream.collect_image_attachments(submission)

    def judge_one(item: tuple[int, dict]) -> tuple[dict, dict]:
        index, rubric = item
        return upstream.judge_rubric(index, rubric, contents, settings["model"], GEMINI_BASE_URL, "",
                                     settings["timeout_per_rubric"], settings["max_retries"], images)

    with ThreadPoolExecutor(settings["rubric_workers"]) as pool:
        judged = list(pool.map(judge_one, enumerate(rubrics)))
    # Upstream turns a judge call that failed every retry into a failed item; that is infrastructure.
    status = "ok" if all(debug["api_exit_code"] == 0 for _, debug in judged) else "infra_error"
    return status, [result for result, _ in judged]


def aggregate(rows: list[dict]) -> dict:
    def macro(group: list[dict]) -> float | None:
        values = [row["normalized_score"] for row in group if row["eligible"]]
        return round(statistics.fmean(values), 6) if values else None

    eligible = sum(row["eligible"] for row in rows)
    return {"tasks": len(rows), "eligible": eligible, "unavailable": len(rows) - eligible, "macro": macro(rows),
            "by_split": {split: macro([row for row in rows if row["split"] == split]) for split in ("main", "easy")}}


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run inside upstream's environment; see `evaluate`.")
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("task_ids", nargs="+")
    args = ap.parse_args()
    settings = json.loads(SETTINGS.read_text(encoding="utf-8"))
    spec = importlib.util.spec_from_file_location("jobbench_upstream_judge", args.data / "job-bench-eval/eval/judge.py")
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    transport = GeminiTransport(settings)
    upstream.get_openai_client = transport.client
    lines = (args.data / "tasks.jsonl").read_text(encoding="utf-8").splitlines()
    tasks = {row["canonical_task_id"]: row for row in map(json.loads, lines)}
    out = args.run / "eval"
    rows = []
    for task_id in args.task_ids:
        rubrics = json.loads((args.data / tasks[task_id]["rubric_path"]).read_text(encoding="utf-8"))["rubrics"]
        record = args.run / task_id / "result.json"
        # A task without a run record failed before producing one: infrastructure.
        submission_status = json.loads(record.read_text(encoding="utf-8"))["status"] if record.is_file() else "infra_error"
        transport.log_path = out / task_id / "calls.jsonl"
        status, results = judge_task(upstream, rubrics, args.run / task_id / "final" / "submission",
                                     submission_status, settings)
        scorecard = upstream.build_scorecard(results)
        eligible = status in {"ok", "model_failure"}
        _write(out / task_id / "rubrics.json", results)
        rows.append({"task_id": task_id, "split": task_id.split("/")[0], "submission_status": submission_status,
                     "status": status, "eligible": eligible,
                     "normalized_score": float(scorecard["normalized_score"]) if eligible else None,
                     "total_score": scorecard["total_score"], "max_score": scorecard["max_score"],
                     "passed_count": scorecard["passed_count"], "rubric_count": len(rubrics)})
        print(f"{task_id}: {status} score={rows[-1]['normalized_score']}", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                                       encoding="utf-8")
    _write(out / "summary.json", aggregate(rows))


if __name__ == "__main__":
    main()
