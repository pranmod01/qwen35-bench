#!/usr/bin/env python
"""Accuracy against where in the study list the tested item sat.

  python -m bench.plot_position [--out plots/position]

Two sources:

1. The final runs bench.plot_decay reads, where the tested item sits at a random position. Trials are binned by
   relative position (N_BINS equal parts of the list, first to last) and pooled over seeds and over list sizes of
   MIN_N or more. Position is `study_position` for paired associates, `metadata.test_position` for token
   recognition, and the earlier of the two probed items for serial order (`first_serial_position`). Image recognition cells from before
   2026-11-11 did not store it; it is recovered by replaying the seeded shuffle (replayed_positions), and a cell
   is used only if the replay reproduces its stored answer key.
   -> position_curves.png, position_table.csv (per list size, so nothing is lost to the pooling)

2. results/recognition_position (configs/recognition_position_*.yaml), where `test_positions` pins the tested item
   to fixed fractions of the list, the same item and foil at every position under one seed.
   -> position_pinned.png, position_pinned.csv
"""
import argparse
import csv
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bench.plot import MODELS, base_model
from bench.plot_decay import MODALITIES, RUN_DIRS, TASKS, modality
from src.metrics import wilson_ci

PINNED_DIR = "results/recognition_position"
N_BINS = 5
MIN_N = 25   # shorter lists put under 5 items in a bin, and most models are at ceiling on them


def dataset_sizes():
    """Items on disk per image dataset: the length of the first shuffle in each single_foil episode."""
    root = Path("memory_datasets")
    return {
        "Brady2008": sum(p.lower().endswith((".jpg", ".png", ".jpeg")) for p in os.listdir(root / "Brady2008Objects")),
        "imagenet": len(json.loads((root / "ImageNet21k_w21_256/concepts.json").read_text())["concepts"]),
        "things": sum(p.is_dir() for p in os.scandir(root / "THINGS/object_images")),
    }


def replayed_positions(result, sizes):
    """Study position of the tested image in each trial of a single_foil cell that did not record it.

    Follows the random draws of AFCRecognitionTask (tasks/recognition.py) without loading an image: the dataset
    shuffle(s), the shuffle of the study list (the tested pair starts at index 0), then the coin for which option
    is correct. That coin is the check: None unless every replayed target equals the stored one.
    """
    cond = result["run"]["condition"]
    n, dataset = cond["n_images"], cond["dataset"]
    random.seed(result["run"]["seed"])
    positions, targets = [], []
    for _ in result["trials"]:
        random.shuffle(list(range(sizes[dataset])))
        if dataset == "things":                 # ThingsDataset shuffles its categories, the task then its n + 1 items
            random.shuffle(list(range(n + 1)))
        order = list(range(n))
        random.shuffle(order)
        positions.append(order.index(0) + 1)
        targets.append(1 if random.random() < 0.5 else 2)
    return positions if targets == [t["target"] for t in result["trials"]] else None


def trial_positions(result, sizes):
    task, trials = result["run"]["task"], result["trials"]
    if task == "paired_associates":
        return [t["study_position"] for t in trials]
    if task == "serial_order":
        # The earlier item; a late one leaves little room for the other, so late bins hold closer (harder) pairs.
        return [t["metadata"]["first_serial_position"] for t in trials]
    if all("metadata" in t for t in trials):
        return [t["metadata"]["test_position"] for t in trials]
    return replayed_positions(result, sizes)


def results_in(run_dir):
    for path in sorted(Path(run_dir).glob("*/*.json")):
        r = json.loads(path.read_text())
        if not r.get("error") and r.get("trials"):
            yield path, r


def load_random(run_dirs):
    """{(task, modality, model, n): [[correct, trials] per bin]} for the random-position runs, bench.plot_decay's cells."""
    sizes = dataset_sizes()
    cells = defaultdict(lambda: [[0, 0] for _ in range(N_BINS)])
    unplaced = []
    for d in run_dirs:
        for path, r in results_in(d):
            task, cond = r["run"]["task"], r["run"]["condition"]
            filt = next((f for t, f, _ in TASKS if t == task), None)
            if filt is None or any(cond.get(k, "") != v for k, v in filt.items()):
                continue
            if cond.get("corruption") or cond.get("baseline") or cond.get("token_pool", "clean") != "clean":
                continue
            n = cond["n_images"]
            if n < MIN_N or "test_position" in cond:
                continue
            positions = trial_positions(r, sizes)
            if positions is None:
                unplaced.append(path.name)
                continue
            cell = cells[(task, modality(cond), base_model(r["run"]["model"]["name"]), n)]
            for trial, position in zip(r["trials"], positions):
                b = min(N_BINS - 1, int((position - 1) * N_BINS / n))
                cell[b][0] += int(trial["correct"] or 0)
                cell[b][1] += 1
    if unplaced:
        print(f"{len(unplaced)} cell(s) left out, replay did not reproduce the stored targets:\n  " + "\n  ".join(unplaced))
    return cells


def load_pinned(run_dir):
    """{(modality, model, n): {fraction: [correct, trials]}} for the pinned-position run."""
    cells = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for _, r in results_in(run_dir):
        cond = r["run"]["condition"]
        if "test_position" not in cond:
            continue
        cell = cells[(modality(cond), base_model(r["run"]["model"]["name"]), cond["n_images"])][cond["test_position"]]
        cell[0] += sum(int(t["correct"] or 0) for t in r["trials"])
        cell[1] += len(r["trials"])
    return cells


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def draw(ax, xs, counts, k, color, marker, label):
    """One model's points with Wilson 95% intervals, nudged sideways so five models' bars do not overprint."""
    pts = [(x, wilson_ci(c, t)) for x, (c, t) in zip(xs, counts) if t]
    if not pts:
        return
    dx = (k - (len(MODELS) - 1) / 2) * 0.012
    x = [p[0] + dx for p in pts]
    acc, lo, hi = zip(*(p[1] for p in pts))
    # Clamped: at 0 or 1 the Wilson bound can miss the estimate by a rounding error, and errorbar refuses negatives.
    yerr = [[max(0.0, a - l) for a, l in zip(acc, lo)], [max(0.0, h - a) for a, h in zip(acc, hi)]]
    ax.errorbar(x, acc, yerr=yerr, color=color,
                marker=marker, markersize=4.5, linewidth=1.3, elinewidth=0.7, capsize=0, alpha=0.9, label=label)


def style(ax, chance):
    if chance is not None:
        ax.axhline(chance, color="#888888", linewidth=0.8, linestyle="--")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0, 1.02)
    ax.grid(True, color="#e6e6e6", linewidth=0.6)
    ax.tick_params(labelsize=8)   # bench.plot's rcParams size ticks for its single-axes figures
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def legend(fig, axes):
    seen = {}
    for ax in axes.flat:
        seen.update(zip(*reversed(ax.get_legend_handles_labels())))
    labels = [label for _, label, _, _ in MODELS if label in seen]
    fig.legend([seen[label] for label in labels], labels, loc="lower center", ncol=len(labels), fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.01))


def plot_random(cells, out):
    mods = [m for m in MODALITIES if any(k[1] == m[0] for k in cells)]
    centers = [(b + 0.5) / N_BINS for b in range(N_BINS)]
    fig, axes = plt.subplots(len(TASKS), len(mods), figsize=(3.6 * len(mods), 2.9 * len(TASKS)), squeeze=False, sharey="row")
    for i, (task, _, title) in enumerate(TASKS):
        for j, (mod, mod_label) in enumerate(mods):
            ax = axes[i][j]
            for k, (model, label, color, marker) in enumerate(MODELS):
                pooled = [[0, 0] for _ in range(N_BINS)]
                for (t, m, mdl, _), bins in cells.items():
                    if (t, m, mdl) == (task, mod, model):
                        for b in range(N_BINS):
                            pooled[b][0] += bins[b][0]
                            pooled[b][1] += bins[b][1]
                draw(ax, centers, pooled, k, color, marker, label)
            # Paired associates' chance is 1/n, under 0.04 for every size pooled here.
            style(ax, None if task == "paired_associates" else 0.5)
            if i == 0:
                ax.set_title(mod_label, fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{title}\naccuracy", fontsize=9)
            if i == len(TASKS) - 1:
                ax.set_xlabel("position in study list (first → last)", fontsize=9)
    legend(fig, axes)
    fig.suptitle(f"Accuracy by position of the tested item, random-position runs pooled over list sizes ≥ {MIN_N} "
                 "(serial order: the earlier of the two items)", fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(out / "position_curves.png", dpi=160)
    plt.close(fig)


def plot_pinned(cells, out):
    mods = [m for m in MODALITIES if any(k[0] == m[0] for k in cells)]
    sizes = {mod: sorted({k[2] for k in cells if k[0] == mod}) for mod, _ in mods}
    ncols = max(len(s) for s in sizes.values())
    fig, axes = plt.subplots(len(mods), ncols, figsize=(3.6 * ncols, 2.9 * len(mods)), squeeze=False, sharey=True)
    for i, (mod, mod_label) in enumerate(mods):
        for j in range(ncols):
            ax = axes[i][j]
            if j >= len(sizes[mod]):
                ax.set_visible(False)
                continue
            n = sizes[mod][j]
            for k, (model, label, color, marker) in enumerate(MODELS):
                by_fraction = cells.get((mod, model, n), {})
                fractions = sorted(by_fraction)
                draw(ax, fractions, [by_fraction[f] for f in fractions], k, color, marker, label)
            style(ax, 0.5)
            ax.set_title(f"{mod_label}, n = {n:,}", fontsize=10)
            ax.set_xlabel("position of the tested item (0 first, 1 last)", fontsize=9)
            if j == 0:
                ax.set_ylabel("2AFC recognition accuracy", fontsize=9)
    legend(fig, axes)
    fig.suptitle("Recognition with the tested item pinned to a position", fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(out / "position_pinned.png", dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="plots/position")
    ap.add_argument("--runs", nargs="+", default=RUN_DIRS)
    ap.add_argument("--pinned", default=PINNED_DIR)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cells = load_random(args.runs)
    write_csv(out / "position_table.csv", ["task", "modality", "model", "n", "bin", "correct", "trials"],
              [(*key, b + 1, c, t) for key, bins in sorted(cells.items()) for b, (c, t) in enumerate(bins)])
    plot_random(cells, out)
    print(f"random-position runs: {len(cells)} cells -> {out}/position_curves.png, position_table.csv")

    pinned = load_pinned(args.pinned)
    if pinned:
        write_csv(out / "position_pinned.csv", ["modality", "model", "n", "test_position", "correct", "trials"],
                  [(*key, f, c, t) for key, by_f in sorted(pinned.items()) for f, (c, t) in sorted(by_f.items())])
        plot_pinned(pinned, out)
        print(f"pinned run: {len(pinned)} cells -> {out}/position_pinned.png, position_pinned.csv")
    else:
        print(f"no pinned-position results in {args.pinned} yet")


if __name__ == "__main__":
    main()
