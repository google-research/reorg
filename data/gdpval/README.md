# GDPval

* **Source:** [`openai/gdpval`](https://huggingface.co/datasets/openai/gdpval) at revision `11e7900c`, 220 tasks.
* **Cohort:** the 185 tasks that come with an expert deliverable (`splits/cohort_185.txt`, in parquet order).
* **Adaptation stream:** 20 tasks, in stream order (`splits/adapt_20.txt`). The other 165 are held out (`splits/heldout_165.txt`). Main GDPval results are reported on all 185 tasks.
* **Worker inputs:** the request (`prompt`) and its reference files. Every action prompt also lists the dataset paths of the expected deliverables (`deliverable_files/<hash>/<name>`) and the worker's finish tool expects files with those names. The expert files themselves and the rubrics are used only for evaluation.
* **Score:** win + tie rate against the expert deliverable, by majority vote of three judges (App. A.5.2).

## Setup

```bash
python3 reorg.py --benchmark gdpval prepare
```

This downloads the snapshot into `downloads/gdpval/` and checks three things:

1. the parquet's sha256;
2. one digest over 510 files: the parquet plus every reference and expert file (`files_digest` in `splits/manifest.json`);
3. that `cohort_185.txt` is exactly the set of tasks with expert deliverables.

## Files

| File | Content |
|---|---|
| `splits/manifest.json` | dataset revision and hashes |
| `splits/cohort_185.txt` | evaluation cohort |
| `splits/adapt_20.txt` | adaptation stream, in order |
| `splits/heldout_165.txt` | held-out tasks |
| `splits/smoke.txt` | two tasks for liveness checks |

## Runtime and evaluation

* **Runtime** (`runtime.py`, settings in `reorg.json`):
  * The paper's GDPval actions are reproduced byte for byte: the system prompt, `build_action_prompt` layout, tools and the four Loop-In phase sessions.
  * The worker gets web search; stakeholders do not.
  * The submission is every file of the final artifact.
* **Evaluation** (`judge/`, App. A.5.2):
  * Gemini 3.1 Pro, Qwen 3.6 27B and Claude Sonnet 4.6 each compare the submission with the expert deliverable in a blinded workspace.
  * The majority preference decides; a split vote is a tie.
  * The score is (wins + ties) / cohort size.
  * Needs `GEMINI_API_KEY`, `VLLM_BASE_URL`, and `VERTEX_PROJECT` with application-default credentials.
