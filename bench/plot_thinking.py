#!/usr/bin/env python
"""Thinking on vs off: one figure per (task, dataset), one column per model size.

Accuracy with thinking off and on, Wilson 95% CIs. Only the study lengths the thinking runs cover are
drawn (they stop at n=500), so both curves span the same range.

Usage: python -m bench.plot_thinking --off results/recognition_single_foil results/serial_order_final \
           results/pam_final --on results/recognition_single_foil_thinking \
           results/serial_order_final_thinking results/pam_final_thinking [--out plots/thinking]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bench.plot import DATASET_LABELS, MODELS, TASKS, base_model, plot_chance
from bench.summarize import load_rows

# Condition is the category here, so the first two categorical slots, validated as a pair
# (dataviz validate_palette.js: CVD dE 24.7); linestyle and marker repeat the distinction.
CONDITIONS = [
    ("off", "Thinking off", "#2a78d6", "o", "-"),
    ("on",  "Thinking on",  "#eb6834", "s", "--"),
]


def panel_labels(sizes):
    """Tick labels for the narrow panels: a size whose next size is under ~2x away goes unlabelled
    (250 next to 500 printed as "250500" at this width); the largest is always labelled."""
    return [f"{x:,}" if i == len(sizes) - 1 or sizes[i + 1] / x >= 2.1 else "" for i, x in enumerate(sizes)]


def set_log_x(ax, sizes):
    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels(panel_labels(sizes))
    ax.minorticks_off()
    ax.set_xlim(min(sizes) * 0.7, max(sizes) * 1.4)
    ax.grid(True, alpha=0.3)


def plot_condition(ax, cells, color, marker, linestyle, label):
    cells = sorted(cells, key=lambda r: r["n_images"])
    xs = [r["n_images"] for r in cells]
    acc = [100 * r["accuracy"] for r in cells]
    lo = [max(0.0, 100 * (r["accuracy"] - r["ci95_low"])) for r in cells]
    hi = [max(0.0, 100 * (r["ci95_high"] - r["accuracy"])) for r in cells]
    ax.errorbar(xs, acc, yerr=[lo, hi], marker=marker, linestyle=linestyle, markersize=7, linewidth=2,
                elinewidth=1, capsize=3, color=color, label=label)


def make_figure(off_rows, on_rows, task, dataset, title, out_dir):
    order = [m for m, *_ in MODELS]
    models = sorted({base_model(r["model"]) for r in on_rows}, key=order.index)
    labels = {m: label for m, label, *_ in MODELS}
    sizes = sorted({r["n_images"] for r in on_rows})
    off_rows = [r for r in off_rows if r["n_images"] in sizes]

    fig, axes = plt.subplots(1, len(models), figsize=(5.6 * len(models), 5.2), squeeze=False)
    for col, model in enumerate(models):
        top = axes[0, col]
        for key, label, color, marker, linestyle in CONDITIONS:
            rows = off_rows if key == "off" else on_rows
            cells = [r for r in rows if base_model(r["model"]) == model]
            if cells:
                plot_condition(top, cells, color, marker, linestyle, label)
        plot_chance(top, [r for r in on_rows if base_model(r["model"]) == model])
        top.set_title(labels[model])
        top.set_ylim(0, 105)
        top.set_yticks([0, 20, 40, 60, 80, 100])
        set_log_x(top, sizes)

        top.set_xlabel("Study sequence length")
        if col == 0:
            top.set_ylabel("Accuracy (%)")

    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    order_idx = sorted(range(len(legend_labels)), key=lambda i: legend_labels[i].startswith("Chance"))
    fig.legend([handles[i] for i in order_idx], [legend_labels[i] for i in order_idx],
               loc="lower center", ncol=3, framealpha=0.9, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"{title} ({DATASET_LABELS[dataset]}): thinking off vs on")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    out = Path(out_dir) / f"thinking_vs_off_{task}_{dataset}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--off", nargs="+", required=True, help="no-thinking run dirs")
    parser.add_argument("--on", nargs="+", required=True, help="thinking run dirs")
    parser.add_argument("--out", default="plots/thinking")
    args = parser.parse_args()

    off_rows, on_rows = load_rows(*args.off), load_rows(*args.on)
    Path(args.out).mkdir(parents=True, exist_ok=True)

    for task, filt, datasets, title in TASKS:
        for dataset in datasets:
            def select(rows):
                return [r for r in rows if r["task"] == task and r["dataset"] == dataset
                        and all(r[k] == v for k, v in filt.items())]
            on_sel = select(on_rows)
            if not on_sel:
                continue
            for r in on_sel:
                if r["seeds"] < 2:
                    print(f"note: {r['model']} {task} {dataset} n={r['n_images']} has {r['seeds']} seed "
                          f"({r['episodes']} episodes)")
            make_figure(select(off_rows), on_sel, task, dataset, title, args.out)


if __name__ == "__main__":
    main()
