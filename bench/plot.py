#!/usr/bin/env python
"""Accuracy vs study length, one figure per (task, dataset), all Qwen3.5 sizes on one axes.

Mirrors the layout of vision-memory's src/plot_error_bars.py (log-x study length, accuracy %,
Wilson 95% CIs, legend below the axes) so the two repos' figures can sit side by side.

Usage: python -m bench.plot results/qwen35_0.8b results/qwen35_2b ... [--out plots]
Pass extra-seed and A/B run dirs too (results/qwen35_9b_pam_seeds, results/pam_layout_ab): cells pool
across dirs, and the A/B gets its own figure.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bench.summarize import load_rows

# Model size is ordinal, so one hue light->dark rather than the original's categorical colors;
# markers are kept as a second cue.
MODELS = [
    ("qwen3.5-0.8b", "Qwen3.5-0.8B", "#86b6ef", "o"),
    ("qwen3.5-2b",   "Qwen3.5-2B",   "#5598e7", "s"),
    ("qwen3.5-4b",   "Qwen3.5-4B",   "#2a78d6", "v"),
    ("qwen3.5-9b",   "Qwen3.5-9B",   "#1c5cab", "^"),
    ("qwen3.5-27b",  "Qwen3.5-27B",  "#0d366b", "D"),
]

# (task, row filter, datasets, title). Restricted to the tasks every model size has (Open Item 24).
TASKS = [
    ("afc_recognition", {"foil_type": "novel", "prompt": "original"}, ["Brady2008", "things"],
     "2AFC — Novel"),
    ("serial_order", {"prompt": "original"}, ["Brady2008", "things"],
     "Serial AFC"),
    ("paired_associates", {"prompt": "sandwich"}, ["Brady2008", "things"],
     "PAM (image-word)"),
]
DATASET_LABELS = {"Brady2008": "Brady2008", "things": "Things", "": ""}

# Serial order split by the correct answer, one figure per split (Open Item 25).
HIT_SPLITS = [
    ("hit1", 'Serial AFC — hit1 (correct answer "1")'),
    ("hit2", 'Serial AFC — hit2 (correct answer "2")'),
]

# PAM layout A/B (configs/pam_layout_ab.yaml): linestyle encodes the layout, as the original repo's
# linestyle encodes dataset. PAM results from before the A/B carry no prompt key, hence "".
LAYOUTS = [
    ("", "original", "-"),
    ("pair_numbered", '"Pair k:" boundary', "--"),
    ("pair_plain", '"Pair:" boundary, no index', ":"),
    ("pair_both", '"Pair k:" + "Word k:"', "-."),
    ("tagged", 'tagged image and word', (0, (3, 1, 1, 1))),
    ("sandwich", 'sandwich', (0, (5, 1))),
]
LAYOUT_MODELS = ["qwen3.5-9b", "qwen3.5-27b"]



def base_model(name):
    """THINGS runs at 256 px are stored as e.g. "qwen3.5-2b-img256"; they are still that model size."""
    return name.removesuffix("-img256")


# Same reference point the original repo plots.
HUMAN = {("afc_recognition", "Brady2008"): (2500, 93.0, "Brady et al. 2008")}

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 11,
})


# Minimum ratio between two labelled ticks on the log axis. Study sizes bunch up at the top end
# (500, 1,000, 1,500, 1,637, and the 2,500 human reference), and their labels overprinted each other.
MIN_LABEL_RATIO = 1.7


def thin_labels(xticks):
    """Label ticks only where they are far enough apart to read; keep the first and the last."""
    keep, last = set(), None
    for x in xticks:
        if last is None or x / last >= MIN_LABEL_RATIO:
            keep.add(x)
            last = x
    keep.add(xticks[-1])          # the largest size is the one worth naming
    # Dropping the second-to-last can leave it crowding the last; re-check that pair.
    ordered = sorted(keep)
    if len(ordered) > 1 and ordered[-1] / ordered[-2] < MIN_LABEL_RATIO:
        keep.discard(ordered[-2])
    return [f"{x:,}" if x in keep else "" for x in xticks]


def apply_axes(ax, sizes, max_x):
    xticks = sorted(set(sizes)) + ([max_x] if max_x > max(sizes) else [])
    ax.set_xscale("log")
    ax.set_xticks(xticks)
    ax.set_xticklabels(thin_labels(xticks))
    ax.minorticks_off()
    ax.set_xlim(0.8, max_x * 1.3)
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Study sequence length")
    ax.set_ylabel("Accuracy (%)")


def plot_chance(ax, rows):
    # 50% for the 2-AFC tasks, but 1/n for paired associates, so draw whatever summarize reports.
    pts = sorted({(r["n_images"], r["chance"]) for r in rows})
    xs, ys = zip(*pts)
    if len(set(ys)) == 1:
        ax.axhline(y=100 * ys[0], color="gray", linestyle="--", alpha=0.5, linewidth=1, label="Chance", zorder=0)
    else:
        ax.plot(xs, [100 * y for y in ys], color="gray", linestyle="--", alpha=0.5, linewidth=1,
                label="Chance (1/n)", zorder=0)


def plot_series(ax, cells, color, marker, linestyle, label, metric="accuracy"):
    # accuracy's CI columns are ci95_low/high; the hit splits prefix theirs with the metric name.
    lo_key, hi_key = ("ci95_low", "ci95_high") if metric == "accuracy" else (f"{metric}_ci95_low", f"{metric}_ci95_high")
    cells = sorted(cells, key=lambda r: r["n_images"])
    xs = [r["n_images"] for r in cells]
    acc = [100 * r[metric] for r in cells]
    # Clamped as in the original repo: at 0% or 100% a Wilson bound can sit a float hair past the point.
    lo = [max(0.0, 100 * (r[metric] - r[lo_key])) for r in cells]
    hi = [max(0.0, 100 * (r[hi_key] - r[metric])) for r in cells]
    ax.errorbar(xs, acc, yerr=[lo, hi], marker=marker, linestyle=linestyle,
                markersize=8, linewidth=2.5, elinewidth=1.2, capsize=3, capthick=1.2,
                color=color, label=label)


def finish(ax, fig, title, out_path):
    ax.set_title(title)
    # matplotlib lists plain lines before errorbar containers; put the models first, as in the original.
    handles, labels = ax.get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda i: not labels[i].startswith("Qwen"))
    ax.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, framealpha=0.9, borderaxespad=0)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.32)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def make_plot(rows, task, dataset, title, out_dir):
    fig, ax = plt.subplots(figsize=(8, 6.5))

    for model, label, color, marker in MODELS:
        cells = [r for r in rows if base_model(r["model"]) == model]
        if cells:
            plot_series(ax, cells, color, marker, "-", label)

    plot_chance(ax, rows)
    sizes = [r["n_images"] for r in rows]
    max_x = max(sizes)
    if (task, dataset) in HUMAN:
        x, y, cite = HUMAN[(task, dataset)]
        ax.plot(x, y, marker="*", markersize=14, color="#222222", linestyle="None",
                label=f"Human ({cite})", zorder=5)
        max_x = x

    apply_axes(ax, sizes, max_x)
    finish(ax, fig, f"{title} ({DATASET_LABELS[dataset]})\nAccuracy vs Sequence Length",
           Path(out_dir) / f"errorbars_{task}_{dataset}.png")


def make_hit_plot(rows, dataset, metric, title, out_dir):
    fig, ax = plt.subplots(figsize=(8, 6.5))
    for model, label, color, marker in MODELS:
        cells = [r for r in rows if base_model(r["model"]) == model]
        if cells:
            plot_series(ax, cells, color, marker, "-", label, metric=metric)
    # An unbiased guesser averages 50% on each split; an "always 1" model sits at 100% hit1 / 0% hit2.
    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5, linewidth=1, label="Chance (unbiased guessing)", zorder=0)
    sizes = [r["n_images"] for r in rows]
    apply_axes(ax, sizes, max(sizes))
    ax.set_ylabel("Hit rate (%)")
    finish(ax, fig, f"{title} ({DATASET_LABELS[dataset]})\nHit Rate vs Sequence Length",
           Path(out_dir) / f"errorbars_serial_order_{metric}_{dataset}.png")


def make_layout_plot(rows, dataset, out_dir):
    fig, ax = plt.subplots(figsize=(8, 6.5))
    style = {m: (label, color, marker) for m, label, color, marker in MODELS}
    for model in LAYOUT_MODELS:
        label, color, marker = style[model]
        for prompt, layout_label, linestyle in LAYOUTS:
            cells = [r for r in rows if base_model(r["model"]) == model and r["prompt"] == prompt]
            if cells:
                plot_series(ax, cells, color, marker, linestyle, f"{label} — {layout_label}")
    plot_chance(ax, rows)
    sizes = [r["n_images"] for r in rows]
    apply_axes(ax, sizes, max(sizes))
    finish(ax, fig, f"PAM layout A/B ({DATASET_LABELS[dataset]})\nAccuracy vs Sequence Length",
           Path(out_dir) / f"errorbars_pam_layout_ab_{dataset}.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--out", default="plots")
    args = parser.parse_args()

    rows = load_rows(*args.run_dirs)
    Path(args.out).mkdir(parents=True, exist_ok=True)

    for task, filt, datasets, title in TASKS:
        for dataset in datasets:
            sel = [r for r in rows if r["task"] == task and r["dataset"] == dataset
                   and all(r[k] == v for k, v in filt.items())]
            if not sel:
                continue
            # A model name outside MODELS is silently left off the figure; this is how every THINGS
            # series went missing when those runs were renamed "-img256".
            known = {m for m, *_ in MODELS}
            for name in sorted({r["model"] for r in sel if base_model(r["model"]) not in known}):
                print(f"warning: {task} {dataset}: model {name!r} matches no plotted series")
            # A cell short of the full two seeds gets a wider CI, but say so rather than rely on that.
            for r in sel:
                if r["seeds"] < 2:
                    print(f"note: {r['model']} {task} {dataset} n={r['n_images']} has "
                          f"{r['seeds']} seed ({r['episodes']} episodes)")
            make_plot(sel, task, dataset, title, args.out)
            if task == "serial_order":
                for metric, hit_title in HIT_SPLITS:
                    make_hit_plot(sel, dataset, metric, hit_title, args.out)

    for dataset in ("Brady2008", "things"):
        sel = [r for r in rows if r["task"] == "paired_associates" and r["dataset"] == dataset
               and base_model(r["model"]) in LAYOUT_MODELS]
        if any(r["prompt"] == "pair_numbered" for r in sel):
            make_layout_plot(sel, dataset, args.out)


if __name__ == "__main__":
    main()
