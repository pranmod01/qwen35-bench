#!/usr/bin/env python
"""Image-corruption figures: the ImageNet tasks on corrupted stimuli (results/imagenet_corruption) against the
clean final runs. Styling comes from bench.plot; cells are pooled over seeds as in bench.plot_tokens.

  plots/corruption/gaussian_<task>.png      one panel per model: accuracy vs study size, one line per Gaussian
                                            noise severity (1-5, light to dark) and the clean run (black, dashed).
  plots/corruption/types_<task>_n<n>.png    one panel per corruption type: accuracy vs severity (0 = clean, 1, 3, 5)
                                            at one study size, one line per model.

A model appears in a figure only when every cell of that grid has a result, so a half-finished job does not draw
a curve that will change. Models with partial grids are listed on stdout. The smallest lists (n=1, or 2 for
serial order; configs/corruption_small_n_imagenet.yaml) are drawn when present and not required: they add a
point on the left without moving the others.

Usage: python3 -m bench.plot_corruption [--out plots/corruption]   (system python3: matplotlib is not in the qwen35 venv)
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bench.plot import MODELS, base_model, plot_series
from bench.plot_tokens import load_cells, select, set_axes
from src import corruption

CORRUPTION_DIR = "results/imagenet_corruption"
# The clean reference for each task: the final ImageNet run and its row filter.
CLEAN = {
    "afc_recognition": ("results/recognition_single_foil", {"foil_type": "novel", "prompt": "original"}),
    "serial_order": ("results/serial_order_final", {"prompt": "original"}),
    "paired_associates": ("results/pam_final", {"prompt": "sandwich"}),
}
TITLES = {"afc_recognition": "2AFC — Novel", "serial_order": "Serial AFC", "paired_associates": "PAM (image-word)"}

# The two sweeps' grids (configs/corruption_{gaussian,types}_imagenet.yaml), plus the smallest list per task.
GAUSSIAN_SIZES = [5, 25, 100, 250, 1000, 2500]
SMALLEST = {"afc_recognition": 1, "serial_order": 2, "paired_associates": 1}
GAUSSIAN_SEVERITIES = [1, 2, 3, 4, 5]
TYPES = [n for n in corruption.NAMES if n != "gaussian_noise"]
TYPES_SEVERITIES = [1, 3, 5]
TYPES_SIZES = [5, 100, 1000]

# Severity is ordered, so one hue from light to dark; the clean run is black and dashed so it reads apart.
SEVERITY_COLORS = {1: "#b9d3f3", 2: "#86b6ef", 3: "#5598e7", 4: "#2a78d6", 5: "#13407a"}
CLEAN_STYLE = {"color": "black", "marker": "o", "linestyle": "--"}


def chance(task, n):
    return 1 / n if task == "paired_associates" else 0.5


def load(task):
    clean_dir, clean_filter = CLEAN[task]
    clean = select(load_cells(clean_dir), task=task, dataset="imagenet", **clean_filter)
    corrupted = select(load_cells(CORRUPTION_DIR), task=task)
    return clean, corrupted


def complete(rows, sizes, severities, **filt):
    """True when every (n, severity) cell of the grid is present among the rows."""
    have = {(r["n_images"], r["severity"]) for r in select(rows, **filt)}
    return all((n, s) in have for n in sizes for s in severities)


def model_order(names):
    return [m for m, *_ in MODELS if m in {base_model(n) for n in names}]


def style(model):
    return next((label, color, marker) for m, label, color, marker in MODELS if m == model)


def plot_chance_line(ax, task, sizes):
    if task == "paired_associates":
        ax.plot(sizes, [100 / n for n in sizes], color="gray", linestyle="--", alpha=0.5, linewidth=1, label="Chance (1/n)", zorder=0)
    else:
        ax.axhline(50, color="gray", linestyle="--", alpha=0.5, linewidth=1, label="Chance", zorder=0)


def gaussian_figure(task, clean, corrupted, out_dir):
    rows = select(corrupted, corruption="gaussian_noise")
    models = model_order({r["model"] for r in rows})
    ready, partial = [], []
    for m in models:
        name = next(r["model"] for r in rows if base_model(r["model"]) == m)
        (ready if complete(rows, GAUSSIAN_SIZES, GAUSSIAN_SEVERITIES, model=name) else partial).append(name)
    if partial:
        print(f"  gaussian {task}: partial grids left out: {', '.join(partial)}")
    if not ready:
        print(f"  gaussian {task}: no complete model grid yet")
        return
    sizes = [SMALLEST[task]] + GAUSSIAN_SIZES
    fig, axes = plt.subplots(1, len(ready), figsize=(6.5 * len(ready), 6), squeeze=False, sharey=True)
    for ax, name in zip(axes[0], ready):
        label, *_ = style(base_model(name))
        clean_cells = [r for r in select(clean, model=name) if r["n_images"] in sizes]
        if clean_cells:
            plot_series(ax, clean_cells, label="Clean", **CLEAN_STYLE)
        for s in GAUSSIAN_SEVERITIES:
            plot_series(ax, select(rows, model=name, severity=s), SEVERITY_COLORS[s], "o", "-", f"Severity {s}")
        plot_chance_line(ax, task, sizes)
        set_axes(ax, sizes)
        ax.set_title(label)
    for ax in axes[0][1:]:
        ax.set_ylabel("")
    fig.suptitle(f"{TITLES[task]} — ImageNet, Gaussian noise", y=1.0)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(7, len(labels)), bbox_to_anchor=(0.5, -0.02), framealpha=0.9)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    save(fig, out_dir / f"gaussian_{task}.png")


def severity_series(ax, cells, color, marker, linestyle, label):
    cells = sorted(cells, key=lambda r: r["severity"])
    xs = [r["severity"] for r in cells]
    acc = [100 * r["accuracy"] for r in cells]
    lo = [max(0.0, 100 * (r["accuracy"] - r["ci95_low"])) for r in cells]
    hi = [max(0.0, 100 * (r["ci95_high"] - r["accuracy"])) for r in cells]
    ax.errorbar(xs, acc, yerr=[lo, hi], marker=marker, linestyle=linestyle, markersize=7, linewidth=2,
                elinewidth=1.2, capsize=3, capthick=1.2, color=color, label=label)


def types_figure(task, clean, corrupted, n, out_dir):
    rows = [r for r in corrupted if r["corruption"] in TYPES and r["n_images"] == n]
    models = model_order({r["model"] for r in rows})
    ready, partial = [], []
    for m in models:
        name = next(r["model"] for r in rows if base_model(r["model"]) == m)
        have = {(r["corruption"], r["severity"]) for r in select(rows, model=name)}
        ok = all((c, s) in have for c in TYPES for s in TYPES_SEVERITIES) and select(clean, model=name, n_images=n)
        (ready if ok else partial).append(name)
    if partial:
        print(f"  types {task} n={n}: partial grids left out: {', '.join(partial)}")
    if not ready:
        print(f"  types {task} n={n}: no complete model grid yet")
        return
    ncol = 5
    nrow = -(-len(TYPES) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 4.4 * nrow), squeeze=False, sharey=True)
    for ax, ctype in zip(axes.flat, TYPES):
        for name in ready:
            label, color, marker = style(base_model(name))
            clean_row = dict(select(clean, model=name, n_images=n)[0], severity=0)
            severity_series(ax, [clean_row] + select(rows, model=name, corruption=ctype), color, marker, "-", label)
        plot_chance_line(ax, task, [n])
        ax.set_title(ctype.replace("_", " "), fontsize=14)
        ax.set_xticks([0] + TYPES_SEVERITIES)
        ax.set_xticklabels(["clean", *TYPES_SEVERITIES])
        ax.set_xlim(-0.5, 5.5)
        ax.set_ylim(0, 105)
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.grid(True, alpha=0.3)
    for ax in axes.flat[len(TYPES):]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("Severity")
    for row in axes:
        row[0].set_ylabel("Accuracy (%)")
    fig.suptitle(f"{TITLES[task]} — ImageNet, n = {n:,}", y=1.0)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), bbox_to_anchor=(0.5, -0.01), framealpha=0.9)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    save(fig, out_dir / f"types_{task}_n{n}.png")


def save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="plots/corruption")
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for task in CLEAN:
        clean, corrupted = load(task)
        if not corrupted:
            print(f"  {task}: no corruption results")
            continue
        gaussian_figure(task, clean, corrupted, out_dir)
        # The types sweep runs recognition and serial order only (configs/corruption_types_imagenet.yaml).
        if task != "paired_associates":
            for n in TYPES_SIZES[1:]:
                types_figure(task, clean, corrupted, n, out_dir)


if __name__ == "__main__":
    main()
