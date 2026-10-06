"""Runs the expanded config: one model loaded at a time, every task run against it."""
import json
import time
import traceback
from collections import Counter
from pathlib import Path

import torch

from bench.config import expand_runs, provenance
from bench.models import build_evaluator
from bench.registry import get_adapter


def result_path(cfg, run):
    return Path(cfg["output_dir"]) / cfg["run_name"] / run.task / run.result_name


def is_done(path):
    """A result counts as done only if it exists, did not fail, and was not skipped.

    A skip is a *claim* that the model could not take the prompt, and those claims have been wrong:
    the capacity probe used to cost four times what the real trial cost, and even a correct probe
    can reject a condition that fits on a larger card. Treating a skip as complete meant the cell was
    never retried and quietly stayed absent from the results (the 9B's THINGS n=500 cells, 2026-09-26).
    """
    if not path.exists():
        return False
    try:
        result = json.loads(path.read_text())
    except json.JSONDecodeError:
        return False
    return result.get("error") is None and not result.get("skipped")


def run_all(cfg, overwrite=False, dry_run=False):
    runs = expand_runs(cfg)
    prov = provenance(cfg)
    print(f"{len(runs)} runs: {len(cfg['models'])} model(s) x {len(cfg['tasks'])} task(s) x {len(cfg['seeds'])} seed(s)")

    if dry_run:
        for run in runs:
            status = "done" if is_done(result_path(cfg, run)) and not overwrite else "todo"
            print(f"  [{status}] {run.task}/{run.result_name}")
        return

    for model_cfg in cfg["models"]:
        model_runs = [r for r in runs if r.model["name"] == model_cfg["name"]]
        pending = [r for r in model_runs if overwrite or not is_done(result_path(cfg, r))]
        if not pending:
            print(f"\n=== {model_cfg['name']}: all {len(model_runs)} results already exist, skipping load")
            continue

        print(f"\n=== {model_cfg['name']}: {len(pending)}/{len(model_runs)} runs to do")
        evaluator = build_evaluator(model_cfg)

        for i, run in enumerate(pending, 1):
            out = result_path(cfg, run)
            print(f"[{i}/{len(pending)}] {run.task} {run.condition} seed={run.seed}", flush=True)
            t0 = time.time()
            for _dev in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(_dev)
            try:
                result = get_adapter(run.task).run(evaluator, run.condition, run.seed)
                error = None
            except Exception:
                result, error = {}, traceback.format_exc()
                # These wrappers no longer pre-check capacity, so an OOM surfaces here; free the
                # cache so the next run in this job does not start on a fragmented GPU. Guarded:
                # after a CUDA fault (not an OOM) this call raises too, which killed the job before
                # the failure was written (2026-09-30, illegal memory access on RTX PRO 6000).
                try:
                    torch.cuda.empty_cache()
                except Exception as cache_error:
                    print(f"  empty_cache failed: {cache_error.__class__.__name__}", flush=True)
                print(f"  FAILED:\n{error}", flush=True)

            if result.get("trials"):
                # How often each answer was given; separates response bias from memory
                answers = [str(t.get("response", t.get("reported"))) for t in result["trials"]]
                result["metrics"]["response_distribution"] = dict(Counter(answers).most_common())

            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({
                "run": {"model": run.model, "task": run.task, "condition": run.condition, "seed": run.seed},
                "config": cfg,
                "provenance": prov,
                "elapsed_s": round(time.time() - t0, 1),
                # Summed across devices: a sharded model's device 0 alone understates it badly.
                "peak_gpu_gb": round(sum(torch.cuda.max_memory_allocated(d)
                                         for d in range(torch.cuda.device_count())) / 1e9, 2),
                "error": error,
                **result,
            }, indent=2))

            if result.get("skipped"):
                print(f"  SKIPPED: {result['skipped']}", flush=True)
            elif result.get("metrics"):
                summary = {k: v for k, v in result["metrics"].items() if isinstance(v, (int, float))}
                print(f"  {summary} ({time.time() - t0:.1f}s)", flush=True)

        del evaluator
        torch.cuda.empty_cache()
