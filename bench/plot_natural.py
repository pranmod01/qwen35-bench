#!/usr/bin/env python
"""Natural-text token items (results/tokens_natural): WikiText spans kept in order (`natural`) or shuffled
(`scrambled`), the two pools sharing token frequencies. One figure per task, one row per item length, one column
per model size, as bench/plot_tokens.recall_vs_recognition.

Recombined foils carry a language prior (an intact span reads better than two halves), so those figures also draw
the `baseline: true` cells, where the tested item was never studied: memory is the gap between a pool's solid line
and its dotted one, not its height above chance. Novel-foil recognition and serial order have no such prior and
draw the two pools alone.

Usage: python3 -m bench.plot_natural [--out plots]   (system python3: matplotlib is not in the qwen35 venv)
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

from bench.plot import MODELS
from bench.plot_tokens import chance_line, finish_grid, load_cells, panel_grid, select, set_axes

NATURAL = "results/tokens_natural"
ITEM_LENGTHS = [8, 64]
# Pool is the category: the validated condition pair of bench/plot_thinking.py. The unstudied baseline keeps its
# pool's color and switches to a dotted line with hollow markers, so the pair never rests on color alone.
POOLS = [
    ("natural", "Natural spans", "#2a78d6", "o", "-"),
    ("scrambled", "Scrambled spans", "#eb6834", "s", "--"),
]

FIGURES = [
    ("afc_recognition", {"foil_type": "novel"}, "natural_recognition_novel", "2AFC — Novel foil, natural vs scrambled text"),
    ("afc_recognition", {"foil_type": "recombined"}, "natural_recognition_recombined",
     "2AFC — Recombined foil, natural vs scrambled text (dotted: item unstudied, the language prior alone)"),
    ("serial_order", {}, "natural_serial_order", "Serial AFC, natural vs scrambled text"),
]


def plot_cells(ax, cells, color, marker, linestyle, label, hollow=False):
    cells = sorted(cells, key=lambda r: r["n_images"])
    xs = [r["n_images"] for r in cells]
    acc = [100 * r["accuracy"] for r in cells]
    lo = [max(0.0, 100 * (r["accuracy"] - r["ci95_low"])) for r in cells]
    hi = [max(0.0, 100 * (r["ci95_high"] - r["accuracy"])) for r in cells]
    ax.errorbar(xs, acc, yerr=[lo, hi], marker=marker, linestyle=linestyle, markersize=7, linewidth=2,
                elinewidth=1, capsize=3, color=color, label=label,
                markerfacecolor="white" if hollow else color, markeredgewidth=1.5)


def task_figure(rows, task, filt, name, title, out_dir):
    studied = [r for r in select(rows, task=task, **filt) if not r.get("baseline")]
    baseline = [r for r in select(rows, task=task, **filt) if r.get("baseline")]
    if not studied:
        print(f"no results for {name}, skipped")
        return
    lengths = [L for L in ITEM_LENGTHS if select(studied, item_length=L)]
    models = [m for m in MODELS if any(r["model"] == m[0] for r in studied)]
    fig, axes = panel_grid(models, width=5.6, rows=len(lengths))
    for i, L in enumerate(lengths):
        for j, (model, label, *_) in enumerate(models):
            ax = axes[i * len(models) + j]
            sizes = []
            for pool, pool_label, color, marker, linestyle in POOLS:
                cells = select(studied, model=model, item_length=L, token_pool=pool)
                base = select(baseline, model=model, item_length=L, token_pool=pool)
                if cells:
                    plot_cells(ax, cells, color, marker, linestyle, pool_label)
                if base:
                    plot_cells(ax, base, color, marker, ":", f"{pool_label}, unstudied item", hollow=True)
                sizes += [r["n_images"] for r in cells + base]
            chance_line(ax)
            set_axes(ax, sizes, decades=True)
            ax.set_ylabel(f"L = {L}\nAccuracy (%)")
            ax.set_title(label if i == 0 else "")
            if i < len(lengths) - 1:
                ax.set_xlabel("")
    finish_grid(fig, axes, title, out_dir / f"{name}.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="plots")
    args = parser.parse_args()
    out_dir = Path(args.out) / "tokens"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_cells(NATURAL)
    for task, filt, name, title in FIGURES:
        task_figure(rows, task, filt, name, title, out_dir)


if __name__ == "__main__":
    main()
