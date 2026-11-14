#!/usr/bin/env python
"""Phase 1 of the interpretability plan: accuracy against list size per (task, modality, model), with a smooth guide.

  python -m bench.plot_decay [--out plots/decay]

Reads the final image runs (results/recognition_single_foil, serial_order_final, pam_final) and the token runs
(results/tokens_full, tokens_pam), pools trials over seeds, and draws each accuracy-vs-size curve. The line
through the points is a logistic in log n between chance and a fitted ceiling:

    acc(n) = chance(n) + (ceiling - chance(n)) / (1 + exp(slope * (ln n - ln n_mid)))

It is a guide for the eye only; no fitted parameter is reported. Chance is 0.5 for the forced-choice tasks and
1/n for paired associates, so it is evaluated per point. A curve gets no line when the model never leaves chance
(ceiling below chance + MIN_CEILING_LIFT) or when the sizes run do not pin the fall down.

Writes <out>/decay_curves.png.
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from bench.plot import MODELS, base_model

RUN_DIRS = ["results/recognition_single_foil", "results/serial_order_final", "results/pam_final",
            "results/tokens_full", "results/tokens_pam"]
# (task, condition filter, title); the same cells bench.plot draws.
TASKS = [
    ("afc_recognition", {"foil_type": "novel", "prompt": "original"}, "2AFC recognition"),
    ("serial_order", {"prompt": "original"}, "Serial order"),
    ("paired_associates", {"prompt": "sandwich"}, "Paired associates"),
]
MODALITIES = [("Brady2008", "Brady"), ("things", "THINGS"), ("imagenet", "ImageNet"), ("tokens:8", "Tokens L8")]
MIN_CEILING_LIFT = 0.15   # a fitted ceiling closer than this to chance means the curve never rose above chance
MIN_POINTS = 4
MAX_LOG_MID_SE = 5.0      # a standard error above this (a factor of e^5 either way) means the grid does not bracket the fall


def chance(task, n):
    return 1.0 / n if task == "paired_associates" else 0.5


def modality(cond):
    if cond["dataset"] == "tokens":
        return f"tokens:{cond['item_length']}"
    return cond["dataset"]


def load_cells(run_dirs):
    """{(task, modality, model): {n: (correct, trials)}} pooled over seeds; conditions as bench.plot filters them."""
    cells = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for d in run_dirs:
        for path in Path(d).glob("*/*.json"):
            r = json.loads(path.read_text())
            if r.get("error") or not r.get("trials"):
                continue
            task, cond, model = r["run"]["task"], r["run"]["condition"], r["run"]["model"]["name"]
            filt = next((f for t, f, _ in TASKS if t == task), None)
            if filt is None or any(cond.get(k, "") != v for k, v in filt.items()):
                continue
            if cond.get("corruption") or cond.get("baseline") or cond.get("token_pool", "clean") != "clean":
                continue
            if task == "paired_associates" and cond["n_images"] == 1:
                continue   # one pair: chance is 1, so the cell says nothing about memory
            cell = cells[(task, modality(cond), base_model(model))][cond["n_images"]]
            cell[0] += sum(int(t["correct"] or 0) for t in r["trials"])
            cell[1] += len(r["trials"])
    return cells


def logistic(logn, ceiling, slope, log_mid, ch):
    return ch + (ceiling - ch) / (1.0 + np.exp(slope * (logn - log_mid)))


def fit(task, points):
    """points: [(n, correct, trials)] -> (ceiling, slope, log_mid) of the guide curve, or None when none is drawn."""
    n = np.array([p[0] for p in points], float)
    acc = np.array([p[1] / p[2] for p in points])
    trials = np.array([p[2] for p in points], float)
    ch = np.array([chance(task, k) for k in n])
    logn = np.log(n)
    if len(points) < MIN_POINTS or (acc - ch).max() < MIN_CEILING_LIFT:
        return None
    # Binomial standard error as weights (floored so a 1.00 cell does not get infinite weight).
    sigma = np.sqrt(np.maximum(acc * (1 - acc), 0.02) / trials)
    f = lambda x, ceiling, slope, log_mid: logistic(x, ceiling, slope, log_mid, ch)
    try:
        (ceiling, slope, log_mid), cov = curve_fit(
            f, logn, acc, p0=[acc.max(), 1.0, logn.mean()], sigma=sigma, absolute_sigma=True,
            bounds=([0.0, 0.0, logn.min() - 5], [1.0, 20.0, logn.max() + 10]), maxfev=20000)
    except RuntimeError:
        return None
    # The covariance is unbounded when the grid never brackets the fall (e.g. one step between two sizes).
    if ceiling - ch.min() < MIN_CEILING_LIFT or math.sqrt(abs(cov[2][2])) > MAX_LOG_MID_SE:
        return None
    return float(ceiling), float(slope), float(log_mid)


def plot_curves(cells, out):
    mods = [m for m in MODALITIES if any(k[1] == m[0] for k in cells)]
    fig, axes = plt.subplots(len(TASKS), len(mods), figsize=(3.6 * len(mods), 2.9 * len(TASKS)), squeeze=False, sharey="row")
    for i, (task, _, title) in enumerate(TASKS):
        for j, (mod, mod_label) in enumerate(mods):
            ax = axes[i][j]
            for model, label, color, marker in MODELS:
                by_n = cells.get((task, mod, model))
                if not by_n:
                    continue
                ns = sorted(n for n, (c, t) in by_n.items() if t)
                acc = [by_n[n][0] / by_n[n][1] for n in ns]
                ax.plot(ns, acc, marker=marker, linestyle="none", color=color, markersize=4.5, label=label, alpha=0.9)
                params = fit(task, [(n, *by_n[n]) for n in ns])
                if params:
                    grid = np.exp(np.linspace(math.log(ns[0]), math.log(ns[-1]), 200))
                    ch = np.array([chance(task, n) for n in grid])
                    ax.plot(grid, logistic(np.log(grid), *params, ch), color=color, linewidth=1.5, alpha=0.8)
            grid = np.exp(np.linspace(0, math.log(max(n for k in cells if k[0] == task and k[1] == mod for n in cells[k])), 100))
            ax.plot(grid, [chance(task, n) for n in grid], color="#888888", linewidth=0.8, linestyle="--")
            ax.set_xscale("log")
            ax.set_ylim(0, 1.02)
            ax.grid(True, color="#e6e6e6", linewidth=0.6)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            if i == 0:
                ax.set_title(mod_label, fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{title}\naccuracy", fontsize=9)
            if i == len(TASKS) - 1:
                ax.set_xlabel("studied items", fontsize=9)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Accuracy against list size (lines: logistic guide; dashed: chance)", fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(out / "decay_curves.png", dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="plots/decay")
    ap.add_argument("--runs", nargs="+", default=RUN_DIRS)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plot_curves(load_cells(args.runs), out)
    print(f"wrote {out / 'decay_curves.png'}")


if __name__ == "__main__":
    main()
