#!/usr/bin/env python
"""Rescore the stored continuous color-memory cells with the current `parse_degrees` (open item 59, 2026-10-29).

Usage (repo root, venv python):
    python scripts/rescore_color_continuous.py [--dry-run] [results/qwen35_*/color_memory ...]

The 0.8B-4B cells (2026-09/10) were scored by the first port's parser, which took the first 1-3 digit run in the
reply; the 9B/27B cells and bench.plot_ported use the corrected one (the last number in [0, 360]). This brings
the stored files in line so bench.summarize, the files and the plot agree. Per trial, `reported`,
`abs_error_degrees` and `correct` are recomputed from `raw_response`; the cell metrics are recomputed the way
bench/tasks/color_memory_eval.py computes them (calculate_color_metrics over the parsed trials,
parse_failure_rate, frac_within_30_degrees) plus the runner's response_distribution. A file is rewritten only when
at least one trial changes, and then its previous metrics and the number of changed trials are kept under
provenance["rescored"]; a second pass therefore changes nothing. Prints old vs new mean absolute error per model
and list length, pooled over seeds.
"""
import argparse
import glob
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.tasks.color_memory_eval import parse_degrees  # noqa: E402
from src.metrics import calculate_color_metrics  # noqa: E402

RESCORED_WITH = "bench.tasks.color_memory_eval.parse_degrees (last number in [0, 360]; 2026-10-29)"


def rescore_trial(trial):
    reported = parse_degrees(trial.get("raw_response"))
    error = None if reported == -1 else abs((reported - trial["target"] + 180) % 360 - 180)
    return {"reported": reported, "abs_error_degrees": error, "correct": None if error is None else int(error <= 30)}


def cell_metrics(trials):
    scored = [t for t in trials if t["reported"] != -1]
    metrics = calculate_color_metrics([t["reported"] for t in scored],
                                      [t["target"] for t in scored]) if scored else {"accuracy": 0}
    metrics["parse_failure_rate"] = sum(t["reported"] == -1 for t in trials) / len(trials)
    metrics["frac_within_30_degrees"] = sum(bool(t["correct"]) for t in trials) / len(trials)
    metrics["response_distribution"] = dict(Counter(str(t["reported"]) for t in trials).most_common())
    return metrics


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dirs", nargs="*", default=sorted(glob.glob("results/qwen35_*/color_memory")))
    parser.add_argument("--dry-run", action="store_true", help="report, write nothing")
    args = parser.parse_args()

    before, after = defaultdict(list), defaultdict(list)
    changed_files = 0
    for d in args.dirs:
        for path in sorted(Path(d).glob("*mode-continuous*.json")):
            data = json.loads(path.read_text())
            trials = data.get("trials") or []
            if data.get("error") or not trials:
                continue
            run = data["run"]
            key = (run["model"]["name"], run["condition"]["n_images"])
            n_changed = 0
            for t in trials:
                before[key].append(t["abs_error_degrees"])
                new = rescore_trial(t)
                if new["reported"] != t["reported"]:
                    n_changed += 1
                t.update(new)
                after[key].append(t["abs_error_degrees"])
            if n_changed == 0:
                continue
            changed_files += 1
            data["provenance"]["rescored"] = {
                "with": RESCORED_WITH, "n_trials_changed": n_changed, "previous_metrics": data["metrics"],
            }
            data["metrics"] = cell_metrics(trials)
            print(f"{path.name}: {n_changed}/{len(trials)} trials changed")
            if not args.dry_run:
                path.write_text(json.dumps(data, indent=2))

    print(f"\n{changed_files} files {'would change' if args.dry_run else 'rewritten'}")
    print(f"{'model':16}{'n':>6}{'old err':>10}{'new err':>10}{'trials':>8}")
    for key in sorted(before, key=lambda k: (k[0], k[1])):
        print(f"{key[0]:16}{key[1]:6}{mean(before[key]):10.1f}{mean(after[key]):10.1f}{len(before[key]):8}")


if __name__ == "__main__":
    main()
