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
"""ReOrg command line.

  python3 reorg.py --benchmark gdpval prepare                       # download the pinned data and verify it
  python3 reorg.py --benchmark gdpval tasks --set adapt             # print a task set: adapt | eval | heldout | smoke
  python3 reorg.py --benchmark gdpval learn --out runs/gdpval/learn_gemini --model gemini_3_1_pro   # Alg. 1
  python3 reorg.py --benchmark gdpval learn-later --library runs/gdpval/learn_gemini/library --out runs/gdpval/later_gemini
  python3 reorg.py --benchmark gdpval infer --library <dir> --model gemini_3_1_pro [--set eval] [--max-rounds 3]
  python3 reorg.py --benchmark gdpval evaluate --run runs/gdpval/<run_id> [--set eval]

Benchmark data is downloaded to --downloads/<benchmark> and runs go to --runs/<benchmark>/<run_id>
(defaults: downloads/ and runs/ in this repository, both git-ignored).
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from reorg.domain import Domain, load_domain, load_module  # noqa: E402
from reorg.files import read_json  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", required=True, help="a folder under data/: gdpval | jobbench | lab")
    ap.add_argument("--downloads", default=str(ROOT / "downloads"), help="where prepare puts benchmark data")
    ap.add_argument("--runs", default=str(ROOT / "runs"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    p = sub.add_parser("tasks")
    p.add_argument("--set", default="eval", choices=Domain.TASK_SETS)
    p.add_argument("--check", action="store_true", help="load every task packet from the prepared data")
    p = sub.add_parser("infer", help="Eq. 1 with a frozen library and planner knowledge")
    p.add_argument("--library", required=True, help="a learned library directory")
    p.add_argument("--model", default="gemini_3_1_pro", help="configs/models/<name>.json; worker, stakeholders and planner")
    p.add_argument("--set", default="eval", choices=Domain.TASK_SETS)
    p.add_argument("--tasks", help="comma-separated task IDs (overrides --set)")
    p.add_argument("--run-id", help="default: a timestamp")
    p.add_argument("--max-rounds", type=int, help="override the reflection budget (0 runs the worker alone)")
    p = sub.add_parser("learn", help="Alg. 1: learn (L, M) from the ordered adaptation stream")
    p.add_argument("--out", required=True, help="output directory; <out>/library is the result; reruns resume")
    p.add_argument("--model", default="gemini_3_1_pro", help="configs/models/<name>.json; every role")
    p.add_argument("--set", default="adapt", choices=Domain.TASK_SETS)
    p = sub.add_parser("learn-later", help="App. A.3: learn later-round precedents for a frozen library")
    p.add_argument("--library", required=True, help="the one-round library, e.g. <learn out>/library")
    p.add_argument("--out", required=True, help="<out>/library is a copy with precedents.md; reruns resume")
    p.add_argument("--model", default="gemini_3_1_pro")
    p.add_argument("--set", default="adapt", choices=Domain.TASK_SETS)
    p = sub.add_parser("evaluate", help="score a run with the benchmark's judge (App. A.5)")
    p.add_argument("--run", required=True, help="runs/<benchmark>/<run_id>")
    p.add_argument("--set", default="eval", choices=Domain.TASK_SETS)
    args = ap.parse_args()

    domain = load_domain(args.benchmark, Path(args.downloads))
    runs = Path(args.runs).resolve()
    if args.cmd == "prepare":
        print(json.dumps(domain.prepare(), indent=2))
    elif args.cmd == "tasks":
        for task_id in domain.task_set(args.set):
            if args.check:
                missing = [path for path in domain.load_task(task_id).references if not path.is_file()]
                if missing:
                    raise SystemExit(f"{task_id}: missing reference files {missing}")
            print(task_id)
    elif args.cmd == "infer":
        from reorg.inference import run_tasks
        from reorg.library import Library
        from reorg.llm import load_profile

        config = read_json(ROOT / "configs" / "inference.json")
        if args.max_rounds is not None:
            config["max_rounds"] = args.max_rounds
        profile = load_profile(args.model)
        run_dir = runs / domain.name / (args.run_id or time.strftime("%Y%m%d_%H%M%S"))
        ids = args.tasks.split(",") if args.tasks else domain.task_set(args.set)
        results = run_tasks(domain, ids, run_dir=run_dir, hooks=load_module(domain.root / "runtime.py"),
                            library=Library(Path(args.library).resolve()), worker=profile,
                            planner_profile=profile, config=config)
        print(json.dumps({"run_dir": str(run_dir), "tasks": len(results),
                          "errors": sum(r.get("status") == "error" for r in results)}, indent=2))
    elif args.cmd == "learn":
        from reorg.continual_learning.learn import run
        from reorg.llm import load_profile

        records = run(domain, domain.task_set(args.set), out_dir=Path(args.out).resolve(),
                      hooks=load_module(domain.root / "runtime.py"), profile=load_profile(args.model),
                      config=read_json(ROOT / "configs" / "learning.json"))
        print(json.dumps({"library": str(Path(args.out) / "library"), "tasks": len(records)}, indent=2))
    elif args.cmd == "learn-later":
        from reorg.continual_learning import later_round
        from reorg.library import Library
        from reorg.llm import load_profile

        later_round.run(domain, domain.task_set(args.set), library=Library(Path(args.library).resolve()),
                        out_dir=Path(args.out).resolve(),
                        hooks=load_module(domain.root / "runtime.py"), profile=load_profile(args.model),
                        config=read_json(ROOT / "configs" / "learning.json"))
        print(json.dumps({"library": str(Path(args.out) / "library")}, indent=2))
    elif args.cmd == "evaluate":
        # GDPval's judge is a package; JobBench's re-runs itself inside the upstream evaluator's env.
        judge_file = domain.root / "judge.py"
        if judge_file.is_file():
            judge = load_module(judge_file)
        else:
            import importlib
            judge = importlib.import_module(f"data.{domain.name}.judge")
        summary = judge.evaluate(Path(args.run).resolve(), domain.task_set(args.set), domain.data)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
