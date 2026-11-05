# Harvey LAB

* **Source:** [`harveyai/harvey-labs`](https://github.com/harveyai/harvey-labs) at commit `e4bf776e`: 1,759 tasks across 26 practice areas.
* **Splits:** LAB ships no train/test split, so we built one.
  * Whole task families (scenarios of one matter) are assigned to test (200 tasks), then valid (200), and the rest to train (1,359).
  * Sampling is stratified by practice area, work type and deliverable file types, with seed `20260717`.
  * `prepare` re-derives this split and checks it.
* **Evaluation cohort:** the 200 test tasks (`splits/test_200.txt`).
* **Adaptation stream:** 20 tasks drawn from train + valid, in stream order (`splits/adapt_20.txt`).
* **Worker inputs:** the instructions, the task's `documents/`, and the names of the files to produce. Only those files are submitted. Rubric criteria are used only for evaluation.
* **Score:** LAB's native criterion-level evaluator; we report the fraction of rubric criteria passed (App. A.5.3).

## Setup

```bash
python3 reorg.py --benchmark lab prepare
```

This clones the repository into `downloads/lab/harvey-labs/` at the pinned commit and builds `downloads/lab/tasks.jsonl` (one rubric-free row per task, with its split). It then checks:

1. the index hash pinned in `splits/manifest.json`;
2. the split sizes;
3. that `test_200.txt` equals the seeded test split;
4. that every adaptation task comes from train + valid.

## Files

| File | Content |
|---|---|
| `splits/manifest.json` | repository, commit, split seed and sizes, index hash |
| `splits/test_200.txt` | evaluation cohort |
| `splits/adapt_20.txt` | adaptation stream, in order |
| `splits/smoke.txt` | two tasks for liveness checks |

## Runtime and evaluation

* **Runtime** (`runtime.py`, `tools.py`, `prompts/`, settings in `reorg.json`):
  * Each action prompt is the shared role and action prompt (`reorg/prompts/`), then the LAB transfer wrapper (`prompts/`), then the task and the incoming contributions.
  * No action has web access, and no Harvey document-skill manuals are staged.
  * A finalization guard limits code execution once the required outputs exist.
  * The submission is the files the task names; it is empty if any is missing.
* **Evaluation** (`judge.py`, App. A.5.3):
  * Harvey's own `evaluation.scoring` grades each criterion with Gemini 3.1 Pro (temperature 0, high thinking, a JSON verdict).
  * Reported: the fraction of rubric criteria passed (pooled) and the per-task mean (macro).
  * Needs `pandoc` and `pip install -e ".[lab-eval]"`.
