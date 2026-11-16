#!/usr/bin/env python
"""Thinking on vs off for token items (open item 84): one figure per task, one row per item length L, one column
per model size, in the style of bench/plot_thinking.py. Off = results/tokens_full and tokens_pam (clean pool), on =
results/tokens_full_thinking and tokens_pam_thinking, drawn over the sizes the thinking runs cover (n <= 500).

The thinking budget is 12,288 tokens and the 4B/9B hit it on every trace at the long lengths, so a cell whose traces
were force-closed on half or more of the trials gets a hollow marker: the answer there follows a truncated trace,
not a finished one. A model's on-curve is drawn only once every cell of its (task, L) grid in the config has landed
(as bench.plot_corruption), so a half-finished job does not draw a curve that will change; partial grids are listed.

Usage: python3 -m bench.plot_thinking_tokens [--out plots/thinking] [--lengths 8 64]   (system python3)
`--lengths` keeps only those rows and adds `_L8_L64` to the file name, so the full figure stays.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import yaml

from bench.plot import MODELS
from bench.plot_tokens import finish_grid, load_cells, panel_grid, plot_chance, select, set_axes
from bench.plot_thinking import CONDITIONS, plot_condition

BUDGET_HIT = 0.5
RUNS = {  # task -> (off dir, on dir, config stem for the on grid, condition filter, chance)
    "afc_recognition": ("results/tokens_full", "results/tokens_full_thinking", "full", {"foil_type": "novel"}, 0.5),
    "serial_order": ("results/tokens_full", "results/tokens_full_thinking", "full", {}, 0.5),
    "paired_associates": ("results/tokens_pam", "results/tokens_pam_thinking", "pam", {}, "1/n"),
}
TITLES = {"afc_recognition": "2AFC — Novel foil", "serial_order": "Serial AFC", "paired_associates": "PAM (cue → word)"}
LENGTHS = [1, 8, 64, 256]


def config_grid(stem, L, task, filt):
    """The n grid a thinking config promises for (task, L), or None when there is no config for that L."""
    path = Path("configs/tokens") / f"{stem}_L{L}_thinking.yaml"
    if not path.exists():
        return None
    for t in yaml.safe_load(path.read_text())["tasks"]:
        if t["task"] == task and all(v in t.get(k + "s", [v]) for k, v in filt.items()):
            return set(t["n_images"])
    return None


def budget_fraction(run_dir):
    """Trial-weighted thinking_frac_budget per (model, task, L, n), seeds pooled."""
    hit, n = defaultdict(float), defaultdict(int)
    for path in Path(run_dir).glob("*/*.json"):
        d = json.loads(path.read_text())
        if d.get("error"):
            continue
        c, m = d["run"]["condition"], d["metrics"]
        key = (d["run"]["model"]["name"], d["run"]["task"], c.get("foil_type"), c["item_length"], c["n_images"])
        # Serial order reports its trial count as "total", so weight by the condition's n_trials.
        hit[key] += m.get("thinking_frac_budget", 0) * c["n_trials"]
        n[key] += c["n_trials"]
    return {k: hit[k] / n[k] for k in n}


def task_figure(task, out_dir, keep_lengths=None):
    off_dir, on_dir, stem, filt, chance = RUNS[task]
    on_all = select(load_cells(on_dir), task=task, **filt)
    if not on_all:
        print(f"no thinking results for {task}, skipped")
        return
    off_all = select(load_cells(off_dir), task=task, token_pool="clean", **filt)
    budget = budget_fraction(on_dir)
    lengths = [L for L in LENGTHS if select(on_all, item_length=L) and (not keep_lengths or L in keep_lengths)]
    suffix = "".join(f"_L{L}" for L in keep_lengths) if keep_lengths else ""
    models = [m for m in MODELS if any(r["model"] == m[0] for r in on_all)]
    fig, axes = panel_grid(models, width=5.6, rows=len(lengths))
    for i, L in enumerate(lengths):
        grid = config_grid(stem, L, task, filt)
        for j, (model, label, *_) in enumerate(models):
            ax = axes[i * len(models) + j]
            on = select(on_all, model=model, item_length=L)
            have = {r["n_images"] for r in on}
            complete = grid is not None and grid <= have
            sizes = sorted(have if complete else (grid or have))
            off = [r for r in select(off_all, model=model, item_length=L) if r["n_images"] in sizes]
            for key, cond_label, color, marker, linestyle in CONDITIONS:
                cells = off if key == "off" else (on if complete else [])
                if cells:
                    plot_condition(ax, cells, color, marker, linestyle, cond_label)
            if complete:
                hit = [r for r in on if budget.get((model, task, r.get("foil_type"), L, r["n_images"]), 0) >= BUDGET_HIT]
                if hit:
                    ax.plot([r["n_images"] for r in hit], [100 * r["accuracy"] for r in hit], linestyle="none",
                            marker="s", markersize=7, markerfacecolor="white", markeredgecolor=CONDITIONS[1][2],
                            markeredgewidth=1.5, label="Thinking on, budget hit on ≥ half the trials", zorder=4)
            else:
                missing = sorted((grid or set()) - have)
                print(f"partial: {model} {task} L={L}: {len(have)}/{len(grid) if grid else '?'} sizes, missing {missing}")
                # Below the chance line, which sits at mid-height.
                ax.text(0.5, 0.25, f"thinking cells pending\n({len(have)}/{len(grid) if grid else '?'} sizes)",
                        transform=ax.transAxes, ha="center", va="center", color="gray", fontsize=10)
            if sizes:
                plot_chance(ax, chance, sizes)
                set_axes(ax, sizes, decades=True)
            else:
                ax.set_xscale("log")
            ax.set_ylabel(f"L = {L}\nAccuracy (%)")
            ax.set_title(label if i == 0 else "")
            if i < len(lengths) - 1:
                ax.set_xlabel("")
    finish_grid(fig, axes, f"{TITLES[task]}, token items: thinking off vs on", out_dir / f"thinking_vs_off_tokens_{task}{suffix}.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="plots/thinking")
    parser.add_argument("--lengths", type=int, nargs="+", choices=LENGTHS, help="item lengths to draw (default: all)")
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for task in RUNS:
        task_figure(task, out_dir, keep_lengths=args.lengths)


if __name__ == "__main__":
    main()
