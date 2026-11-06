#!/usr/bin/env python
"""Figures for the three ported tasks: MST, color memory and the associative-inference layout A/B.

Usage: python -m bench.plot_ported [--mst-color results/qwen35_0.8b ...] [--assoc results/assoc_inf_ab]
                                   [--reasoned results/assoc_inf_reasoned] [--out plots/ported]

  mst_accuracy.png              three-way old/similar/new accuracy vs study length, chance 1/3
  mst_ldi.png                   lure discrimination index, p(similar | lure) - p(similar | foil), chance 0
  color_named_accuracy.png      named color recall, chance 1/6
  color_continuous_error.png    mean absolute circular error in degrees; 90 is uniform guessing. Every trial is
                                rescored here from raw_response with the current parse_degrees (the stored 0.8B-4B
                                cells were rescored in place the same way on 2026-10-29 by
                                scripts/rescore_color_continuous.py, so this is now a no-op safeguard); parse
                                failures are left out of the mean and counted on stderr
  assoc_inf_ab_{image,word}.png the A/B grid: layout x model panels, one line per probe, x = chains studied
  assoc_inf_by_model.png        one panel per probe, one line per model size, pooled over the three layouts and
                                both pair types (300 trials a point) -- the size trend the per-cell grid hides
  assoc_inf_reasoned.png        standard scoring vs reason-then-choose (results/assoc_inf_reasoned: the 2B/4B
                                reply freely for up to 512 tokens and the forced choice is read with that reply
                                in context; same episodes, seed 0), pooled the same way, at 2 and 5 chains

Seeds pool into one cell, as in bench.summarize. Model colors and axes come from bench.plot so these figures
read like the main ones; the probe colors are the first three categorical slots of the dataviz reference palette
(adjacent pairs validated there; no node on this machine to rerun the validator).
"""
import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bench.plot import MODELS, apply_axes, base_model, finish
from bench.tasks.color_memory_eval import parse_degrees
from src.metrics import wilson_ci

PROBES = [
    ("direct_ab", "Direct A-B", "#2a78d6", "o", "-"),
    ("direct_bc", "Direct B-C", "#eb6834", "s", "--"),
    ("inference", "Inference A-C", "#1baf7a", "^", ":"),
]
PROMPT_LABELS = {"original": "original", "cue_turn": "cue in its own turn", "labeled": "labeled options"}
PROBE_TITLES = {"direct_ab": "Direct A-B (studied pair)", "direct_bc": "Direct B-C (studied pair)",
                "inference": "Inference A-C (never studied)"}
SCORING = [("standard", "Standard: choice read after a 100-token reply", "#2a78d6", "o", "-"),
           ("reasoned", "Reason-then-choose: choice read after a 512-token reply", "#eb6834", "s", "--")]


def load_cells(run_dirs, task):
    """{(model, condition-without-n_trials): [trials pooled over seeds]}."""
    cells = defaultdict(list)
    for d in run_dirs:
        for path in sorted(Path(d, task).glob("*.json")):
            r = json.loads(path.read_text())
            if r.get("error") or r.get("skipped"):
                continue
            cond = {k: v for k, v in r["run"]["condition"].items() if k != "n_trials"}
            cells[(r["run"]["model"]["name"], json.dumps(cond, sort_keys=True))].extend(r["trials"])
    return [(model, json.loads(cond), trials) for (model, cond), trials in cells.items()]


def series(ax, points, color, marker, linestyle, label, panel=False):
    """points: [(x, y, lo, hi)] in display units. Mark sizes as bench.plot.plot_series, or as
    bench.plot_thinking.plot_condition for the narrower panels of a grid."""
    points = sorted(points)
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    lo = [max(0.0, p[1] - p[2]) for p in points]
    hi = [max(0.0, p[3] - p[1]) for p in points]
    size = dict(markersize=7, linewidth=2, elinewidth=1) if panel else dict(markersize=8, linewidth=2.5, elinewidth=1.2, capthick=1.2)
    ax.errorbar(xs, ys, yerr=[lo, hi], marker=marker, linestyle=linestyle, capsize=3, color=color, label=label, **size)


def by_model(cells, point):
    """{model: [(n, y, lo, hi)]} with `point(trials)` -> (y, lo, hi), skipping cells it returns None for."""
    out = defaultdict(list)
    for model, cond, trials in cells:
        p = point(trials)
        if p is not None:
            out[base_model(model)].append((cond["n_images"], *p))
    return out


def model_figure(points_by_model, title, ylabel, chance, chance_label, ylim, out_path):
    fig, ax = plt.subplots(figsize=(8, 6.5))
    for model, label, color, marker in MODELS:
        if points_by_model.get(model):
            series(ax, points_by_model[model], color, marker, "-", label)
    ax.axhline(y=chance, color="gray", linestyle="--", alpha=0.5, linewidth=1, label=chance_label, zorder=0)
    sizes = [n for pts in points_by_model.values() for n, *_ in pts]
    apply_axes(ax, sizes, max(sizes))
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    if ylim[1] <= 1.1:            # apply_axes sets percentage ticks; LDI is a difference of rates
        ax.set_yticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0])
    elif ylim[1] > 105:
        ax.set_yticks(range(0, int(ylim[1]) + 1, 30))
    finish(ax, fig, title, out_path)


def pct(k, n):
    p, lo, hi = wilson_ci(k, n)
    return 100 * p, 100 * lo, 100 * hi


def mst_accuracy(trials):
    return pct(sum(int(t["correct"] or 0) for t in trials), len(trials))


def mst_ldi(trials):
    def rate(kind):
        items = [t for t in trials if t["type"] == kind]
        return (sum(str(t["reported"]).lower() == "similar" for t in items) / len(items), len(items)) if items else (None, 0)
    (p1, n1), (p2, n2) = rate("lure"), rate("foil")
    if not n1 or not n2:
        return None
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    ldi = p1 - p2
    return ldi, ldi - 1.96 * se, ldi + 1.96 * se


def color_error(trials):
    """Mean absolute circular error after rescoring from raw_response; None if nothing parses."""
    errors, failures = [], 0
    for t in trials:
        reported = parse_degrees(t.get("raw_response"))
        if reported == -1:
            failures += 1
            continue
        errors.append(abs((reported - t["target"] + 180) % 360 - 180))
    if failures:
        print(f"note: {failures}/{len(trials)} continuous color trials did not parse", file=sys.stderr)
    if not errors:
        return None
    mean = sum(errors) / len(errors)
    sd = math.sqrt(sum((e - mean) ** 2 for e in errors) / max(1, len(errors) - 1))
    half = 1.96 * sd / math.sqrt(len(errors))
    return mean, mean - half, mean + half


def chains_axis(ax, chains):
    ax.set_xscale("log")
    ax.set_xticks(chains)
    ax.set_xticklabels([str(c) for c in chains])
    ax.minorticks_off()
    ax.set_xlim(min(chains) * 0.7, max(chains) * 1.4)
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.grid(True, alpha=0.3)


def grid_legend(fig, axes, title, out_path, bottom=0.06):
    handles, labels = axes[0][0].get_legend_handles_labels()
    order_idx = sorted(range(len(labels)), key=lambda i: labels[i].startswith("Chance"))
    fig.legend([handles[i] for i in order_idx], [labels[i] for i in order_idx],
               loc="lower center", ncol=min(4, len(labels)), framealpha=0.9, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(title)
    fig.tight_layout(rect=(0, bottom, 1, 1))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def pooled(cells, n_key="n_chains", **match):
    """[(n, acc%, lo, hi)] pooling every cell whose condition matches `match`, over everything else."""
    k = defaultdict(lambda: [0, 0])
    for model, cond, trials in cells:
        if all(cond.get(key) == val for key, val in match.items() if key != "model") and \
                (match.get("model") is None or base_model(model) == match["model"]):
            k[cond[n_key]][0] += sum(int(t["correct"] or 0) for t in trials)
            k[cond[n_key]][1] += len(trials)
    return [(n, *pct(c, t)) for n, (c, t) in k.items() if t]


def assoc_figure(cells, pair_type, out_path):
    order = [m for m, *_ in MODELS]
    models = sorted({base_model(c[0]) for c in cells}, key=order.index)
    prompts = [p for p in PROMPT_LABELS if any(c[1]["prompt"] == p for c in cells)]
    fig, axes = plt.subplots(len(prompts), len(models), figsize=(4.2 * len(models), 5.2 * len(prompts)),
                             sharex=True, sharey=True, squeeze=False)
    label_of = dict((m, l) for m, l, *_ in MODELS)
    for i, prompt in enumerate(prompts):
        for j, model in enumerate(models):
            ax = axes[i][j]
            for probe, plabel, color, marker, ls in PROBES:
                pts = [(c["n_chains"], *pct(sum(int(t["correct"] or 0) for t in trials), len(trials)))
                       for m, c, trials in cells
                       if base_model(m) == model and c["prompt"] == prompt and c["pair_type"] == pair_type and c["probe"] == probe]
                if pts:
                    series(ax, pts, color, marker, ls, plabel, panel=True)
            ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5, linewidth=1, label="Chance", zorder=0)
            chains_axis(ax, sorted({c["n_chains"] for _, c, _ in cells}))
            ax.set_title(f"{label_of[model]}, {PROMPT_LABELS[prompt]}")
            if i == len(prompts) - 1:
                ax.set_xlabel("Chains studied")
            if j == 0:
                ax.set_ylabel("Accuracy (%)")
    grid_legend(fig, axes, f"Associative inference (Brady2008): layout A/B, {pair_type} pairs", out_path, bottom=0.04)


def assoc_by_model_figure(cells, out_path):
    """One panel per probe, one line per model size, pooled over layouts and pair types."""
    probes = [p for p in PROBE_TITLES if any(c[1]["probe"] == p for c in cells)]
    fig, axes = plt.subplots(1, len(probes), figsize=(4.2 * len(probes), 5.2), sharey=True, squeeze=False)
    for ax, probe in zip(axes[0], probes):
        for model, label, color, marker in MODELS:
            pts = pooled(cells, model=model, probe=probe)
            if pts:
                series(ax, pts, color, marker, "-", label, panel=True)
        ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5, linewidth=1, label="Chance", zorder=0)
        chains_axis(ax, sorted({c["n_chains"] for _, c, _ in cells}))
        ax.set_title(PROBE_TITLES[probe])
        ax.set_xlabel("Chains studied")
    axes[0][0].set_ylabel("Accuracy (%)")
    grid_legend(fig, axes, "Associative inference (Brady2008) by model size, pooled over layouts and pair types",
                out_path, bottom=0.1)


def assoc_reasoned_figure(standard, reasoned, out_path):
    """Standard vs reason-then-choose, the thinking-figure layout: a column per probe, a row per model."""
    chains = sorted({c["n_chains"] for _, c, _ in reasoned})
    standard = [c for c in standard if c[1]["n_chains"] in chains]
    order = [m for m, *_ in MODELS]
    models = sorted({base_model(c[0]) for c in reasoned}, key=order.index)
    probes = [p for p in PROBE_TITLES if any(c[1]["probe"] == p for c in reasoned)]
    label_of = dict((m, l) for m, l, *_ in MODELS)
    fig, axes = plt.subplots(len(models), len(probes), figsize=(4.2 * len(probes), 5.2 * len(models)),
                             sharex=True, sharey=True, squeeze=False)
    for i, model in enumerate(models):
        for j, probe in enumerate(probes):
            ax = axes[i][j]
            for key, label, color, marker, ls in SCORING:
                pts = pooled(standard if key == "standard" else reasoned, model=model, probe=probe)
                if pts:
                    series(ax, pts, color, marker, ls, label, panel=True)
            ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5, linewidth=1, label="Chance", zorder=0)
            chains_axis(ax, chains)
            ax.set_title(PROBE_TITLES[probe] if i == 0 else "")
            if i == len(models) - 1:
                ax.set_xlabel("Chains studied")
            if j == 0:
                ax.set_ylabel(f"{label_of[model]}\nAccuracy (%)")
    grid_legend(fig, axes, "Associative inference (Brady2008): standard vs reason-then-choose scoring,\n"
                "pooled over layouts and pair types", out_path, bottom=0.08)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mst-color", nargs="+", default=[f"results/qwen35_{m}" for m in ("0.8b", "2b", "4b", "9b", "27b")])
    parser.add_argument("--assoc", nargs="+", default=["results/assoc_inf_ab"])
    parser.add_argument("--reasoned", nargs="+", default=["results/assoc_inf_reasoned"])
    parser.add_argument("--out", default="plots/ported")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    mst = load_cells(args.mst_color, "mst")
    model_figure(by_model(mst, mst_accuracy), "MST — Old / Similar / New (MST)\nAccuracy vs Sequence Length",
                 "Accuracy (%)", 100 / 3, "Chance (1/3)", (0, 105), out / "mst_accuracy.png")
    model_figure(by_model(mst, mst_ldi), "MST — Lure Discrimination (MST)\nLDI vs Sequence Length",
                 "LDI: p(similar | lure) − p(similar | foil)", 0, "Chance (0)", (-0.3, 1.05), out / "mst_ldi.png")

    color = load_cells(args.mst_color, "color_memory")
    named = [c for c in color if c[1]["mode"] == "named"]
    model_figure(by_model(named, mst_accuracy), "Color Memory — Named, 6 colors (Brady2013)\nAccuracy vs Sequence Length",
                 "Accuracy (%)", 100 / 6, "Chance (1/6)", (0, 105), out / "color_named_accuracy.png")
    continuous = [c for c in color if c[1]["mode"] == "continuous"]
    model_figure(by_model(continuous, color_error), "Color Memory — Continuous, color wheel (Brady2013)\nMean Absolute Error vs Sequence Length",
                 "Mean absolute error (degrees)", 90, "Chance (90°)", (0, 125), out / "color_continuous_error.png")

    assoc = load_cells(args.assoc, "associative_inference")
    for pair_type in sorted({c[1]["pair_type"] for c in assoc}):
        assoc_figure(assoc, pair_type, out / f"assoc_inf_ab_{pair_type}.png")
    assoc_by_model_figure(assoc, out / "assoc_inf_by_model.png")
    reasoned = load_cells(args.reasoned, "associative_inference")
    if reasoned:
        assoc_reasoned_figure(assoc, reasoned, out / "assoc_inf_reasoned.png")


if __name__ == "__main__":
    main()
