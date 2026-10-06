#!/usr/bin/env python
"""Pool a run's results across seeds into one row per (model, task, condition).

Usage: python -m bench.summarize results/<run_name> [results/<other_run> ...] [--csv out.csv]
"""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from src.metrics import wilson_ci

UNPARSED = {"-1", ""}

# Flag thresholds, fixed before the 2026-09-22 rerun so a bad cell is reported rather than fixed
# after the fact. LOW_MASS: the forced choice was read off the tail of the distribution on too
# many trials. BIAS: one answer dominates, which for a choice task means the accuracy is not
# evidence of memory (expected where the model is at chance).
LOW_MASS_THRESHOLD = 0.9
LOW_MASS_MAX_FRAC = 0.10
BIAS_MAX_FRAC = 0.80
# PREV_SWAP (paired associates, fixed 2026-09-28 before the layout A/B): most swapped words come from
# the pair studied just before the target, i.e. the layout bound words to the wrong image. A random
# swap lands there about 1/n of the time. Needs trials with `swap_offset`, so older runs never flag.
PREV_SWAP_MIN_SWAPS = 10
PREV_SWAP_MIN_FRAC = 0.5
CHOICE_TASKS = {"afc_recognition", "serial_order", "associative_inference", "mst", "color_memory"}


def chance(task, cond):
    # Paired associates asks for a free word, but models almost always answer with one of the studied
    # words, so the relevant baseline is guessing among them: 1/n (and n=1 is trivially 100%).
    if task == "paired_associates":
        return 1.0 / cond["n_images"]
    if task == "mst":
        return 1.0 / 3       # old | similar | new
    if task == "color_memory":
        # Named: one of six colors. Continuous: the accuracy pooled here is the per-trial "within 30
        # degrees" rate, so chance is the 60-degree window out of 360 -- not the circular-error
        # accuracy in the result file's own metrics, which is 0.5 under random responding.
        return 1.0 / 6
    return 0.5


def load_rows(*run_dirs):
    # Several dirs pool into one row per cell, e.g. a run plus a later run of extra seeds.
    groups = defaultdict(list)
    for path in sorted(p for d in run_dirs for p in Path(d).glob("*/*.json")):
        d = json.loads(path.read_text())
        cond = {k: v for k, v in d["run"]["condition"].items() if k != "n_trials"}
        key = (d["run"]["model"]["name"], d["run"]["task"], json.dumps(cond, sort_keys=True))
        groups[key].append(d)

    rows = []
    for (model, task, cond_json), results in groups.items():
        cond = json.loads(cond_json)
        trials = [t for r in results for t in r.get("trials", [])]
        errors = sum(r.get("error") is not None for r in results)
        skipped = sum(bool(r.get("skipped")) for r in results)
        n = len(trials)
        k = sum(int(t["correct"] or 0) for t in trials)
        acc, lo, hi = wilson_ci(k, n) if n else (float("nan"),) * 3

        answers = Counter(str(t.get("response", t.get("reported"))) for t in trials)
        top_answer, top_count = answers.most_common(1)[0] if answers else ("", 0)

        masses = [t["choice_mass"] for t in trials if "choice_mass" in t]
        low_mass_frac = sum(m < LOW_MASS_THRESHOLD for m in masses) / len(masses) if masses else None
        flags = []
        if low_mass_frac is not None and low_mass_frac > LOW_MASS_MAX_FRAC:
            flags.append("LOW_MASS")
        # A continuous color report is not a choice among options, so answer bias does not apply.
        is_choice = task in CHOICE_TASKS and cond.get("mode") != "continuous"
        if is_choice and n and top_count / n > BIAS_MAX_FRAC:
            flags.append("BIAS")
        offsets = [t["swap_offset"] for t in trials if t.get("swap_offset") is not None]
        if len(offsets) >= PREV_SWAP_MIN_SWAPS and sum(o == -1 for o in offsets) / len(offsets) > PREV_SWAP_MIN_FRAC:
            flags.append("PREV_SWAP")

        # Serial order: accuracy split by the correct answer. A model that always says "1" scores ~50%
        # overall but hit1 = 1.00, hit2 = 0.00, so accuracy alone cannot separate order memory from bias.
        hits = {}
        for digit in (1, 2):
            sub = [t for t in trials if t.get("target") == digit] if task == "serial_order" else []
            h = wilson_ci(sum(int(t["correct"] or 0) for t in sub), len(sub)) if sub else (None,) * 3
            hits.update({f"hit{digit}": h[0], f"hit{digit}_ci95_low": h[1], f"hit{digit}_ci95_high": h[2],
                         f"hit{digit}_n": len(sub)})

        rows.append({
            "model": model,
            "task": task,
            "dataset": cond.get("dataset", ""),
            "foil_type": cond.get("foil_type", ""),
            "prompt": cond.get("prompt", ""),
            "mode": cond.get("mode", ""),      # color memory: named | continuous
            "pair_type": cond.get("pair_type", ""),  # associative inference: image | word
            "probe": cond.get("probe", ""),          # associative inference: inference | direct_ab | direct_bc
            "n_images": cond.get("n_images"),
            "seeds": len(results),
            "episodes": n,
            "accuracy": round(acc, 3),
            "ci95_low": round(lo, 3),
            "ci95_high": round(hi, 3),
            "chance": round(chance(task, cond), 3),
            # small tolerance: wilson_ci's lower bound for k=0 is a float hair above zero
            "above_chance": lo > chance(task, cond) + 1e-9 if n else None,
            "parse_failures": sum(answers[u] for u in UNPARSED),
            # For choice tasks, a model that always gives one answer scores chance without remembering anything
            "most_common_answer": top_answer,
            "most_common_frac": round(top_count / n, 2) if n else None,
            "low_mass_frac": round(low_mass_frac, 2) if low_mass_frac is not None else "",
            "flags": ",".join(flags),
            "errors": errors,
            "skipped": skipped,
            **hits,
        })
    return sorted(rows, key=lambda r: (r["model"], r["task"], r["dataset"], r["foil_type"], r["n_images"], r["prompt"]))


def print_simple(rows):
    """Raw accuracy per cell, for reading the headline before the diagnostics."""
    models = sorted({r["model"] for r in rows})
    datasets = sorted({r["dataset"] for r in rows})
    by_task = defaultdict(dict)
    for r in rows:
        by_task[r["task"]][(r["dataset"], r["model"], r["n_images"])] = r["accuracy"]

    for task, cells in by_task.items():
        sizes = sorted({n for _, _, n in cells})
        header = [f"{m.split('-')[-1]} {d[:6]}" for d in datasets for m in models]
        width = max(12, max(len(h) for h in header) + 1)
        chance = {"paired_associates": "1/n", "mst": "0.33",
                  "color_memory": "0.17"}.get(task, "0.50")
        print(f"### {task} (chance {chance})")
        print(f"{'n':>5} | " + " | ".join(f"{h:>{width}}" for h in header))
        for n in sizes:
            values = [cells.get((d, m, n)) for d in datasets for m in models]
            print(f"{n:>5} | " + " | ".join(f"{v if v is not None else '-':>{width}}" for v in values))
        print()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", help="one or more run dirs; cells are pooled across them")
    parser.add_argument("--csv", help="also write the table to this CSV path")
    parser.add_argument("--simple", action="store_true",
                        help="accuracy only, one column per model/dataset (flags stay in --csv)")
    args = parser.parse_args()

    rows = load_rows(*args.run_dirs)
    if args.simple:
        print_simple(rows)
        if not args.csv:
            return
    cols = ["model", "task", "dataset", "n_images", "prompt", "pair_type", "probe", "episodes", "accuracy", "ci95_low", "ci95_high", "chance",
            "above_chance", "most_common_answer", "most_common_frac", "low_mass_frac", "flags", "errors"]
    print(" | ".join(cols))
    for r in rows:
        print(" | ".join(str(r[c]) for c in cols))

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
