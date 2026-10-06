#!/usr/bin/env python
"""Entry point: python run_bench.py configs/qwen35_0.8b.yaml

Run from the repo root, so tasks find memory_datasets/.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from bench.config import load_config
from bench.runner import run_all


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="path to a YAML config, e.g. configs/qwen35_0.8b.yaml")
    parser.add_argument("--overwrite", action="store_true", help="rerun conditions that already have result files")
    parser.add_argument("--dry-run", action="store_true", help="list the runs without loading a model")
    parser.add_argument("--task", action="append", help="only run this task (repeatable); default is every task in the config")
    parser.add_argument("--model", action="append", help="only run this model name (repeatable); default is every model in the config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.task:
        # One task per job: the 4B's three tasks together do not fit the 12h partition limit.
        # Output paths are per task, so the jobs write into the same run_name side by side.
        kept = [t for t in cfg["tasks"] if t["task"] in args.task]
        missing = set(args.task) - {t["task"] for t in cfg["tasks"]}
        if missing:
            parser.error(f"config has no task(s): {', '.join(sorted(missing))}")
        cfg["tasks"] = kept
    if args.model:
        missing = set(args.model) - {m["name"] for m in cfg["models"]}
        if missing:
            parser.error(f"config has no model(s): {', '.join(sorted(missing))}")
        cfg["models"] = [m for m in cfg["models"] if m["name"] in args.model]

    run_all(cfg, overwrite=args.overwrite, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
