# JobBench

* **Source:** [`JobBench/job-bench`](https://huggingface.co/datasets/JobBench/job-bench) at revision `934d4c93`: 65 main and 63 easy tasks.
* **Task IDs:** `main/<task>` or `easy/<task>`. The prefix is required because 45 raw task IDs occur in both subsets.
* **Adaptation stream:** 20 tasks (10 main + 10 easy), in stream order (`splits/adapt_20.txt`). The other 108 are held out (`splits/heldout_108.txt`). Results are reported on all 128 tasks.
* **Worker inputs:** the request and the visible reference files. Files under a task's `files_required_to_search/` are hidden; only the guarded web search on main tasks may return them. Rubrics are used only for evaluation.
* **Score:** JobBench's native rubric evaluator, averaged over main and easy tasks (App. A.5.3).

## Setup

```bash
python3 reorg.py --benchmark jobbench prepare
```

This downloads the snapshot into `downloads/jobbench/raw/` (985 files). It then writes the split views `downloads/jobbench/{main,easy}` and the task index `downloads/jobbench/tasks.jsonl`, and verifies the index against the hash pinned in `splits/manifest.json`.

## Files

| File | Content |
|---|---|
| `splits/manifest.json` | dataset revision, file patterns, expected counts and hashes |
| `splits/all_128.txt` | evaluation cohort |
| `splits/adapt_20.txt` | adaptation stream, in order |
| `splits/heldout_108.txt` | held-out tasks |
| `splits/smoke.txt` | one main and one easy task for liveness checks |

## Runtime and evaluation

* **Runtime** (`runtime.py`, `tools.py`, `prompts/`, settings in `reorg.json`):
  * Each action prompt is the shared role and action prompt (`reorg/prompts/`), then the JobBench transfer wrapper (`prompts/`), then the task and the incoming contributions.
  * Workers on main tasks get guarded web search, fetch and download with a retrieval budget; easy tasks have no web.
  * Evidence a worker retrieves is passed to later actions.
  * The submission is every file of the final artifact; workers keep helpers in `scratch/`.
* **Evaluation** (`judge.py`, App. A.5.3):
  * The pinned upstream evaluator (`Job-Bench/job-bench-eval` @ `007b8064`) runs in its own `uv` environment with a Gemini 3.5 Flash transport.
  * Settings: high thinking, temperature 0 (the value upstream sends).
  * A JSON-format guard sentence is added to the judge's system prompt, and extracted contents are capped at 150,000 characters.
  * The score is the mean task score over the 128 tasks.
