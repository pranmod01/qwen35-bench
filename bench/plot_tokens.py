#!/usr/bin/env python
"""Token-item figures, and token items overlaid on the image results.

Token runs vary item length L (tokens per item), which bench.summarize.load_rows does not report, so cells are
pooled here per (model, task, condition) straight from the result files. Styling comes from bench.plot.

  plots/tokens/   one figure per measure, one column per item length L, all model sizes on each axes (the layout
                  of bench/plot_thinking.py): recognition (novel and recombined foils), serial order, paired
                  associates (token cue -> word), and the cued-recall probe; recall vs recognition (row per L,
                  column per model); the labeled A/B; the answer-format probe (digit vs copied item vs recall).
                  token items at L=64 against Brady and THINGS-256 images (~66 tokens per item for all three).

Usage: python3 -m bench.plot_tokens [--out plots]   (system python3: matplotlib is not in the qwen35 venv)
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bench.plot import DATASET_LABELS, MODELS, apply_axes, base_model, finish, plot_series
from src.metrics import wilson_ci

TOKENS_FULL = "results/tokens_full"
TOKENS_PAM = "results/tokens_pam"
RECALL = "results/tokens_recall_probe"
AB_LABELED = "results/tokens_ab_labeled"
COPY_PROBE = "results/tokens_copy_probe"
# Per task: the image run to set token items against, its row filter, the token run and its filter, and the chance
# level (a fraction, or "1/n" for paired associates, where guessing among the studied words is the baseline).
MATCHED_TASKS = {
    "afc_recognition": {"image_dir": "results/recognition_single_foil", "image_filter": {"foil_type": "novel", "prompt": "original"},
                        "token_dir": TOKENS_FULL, "token_filter": {"foil_type": "novel"}, "chance": 0.5},
    "serial_order": {"image_dir": "results/serial_order_final", "image_filter": {"prompt": "original"},
                     "token_dir": TOKENS_FULL, "token_filter": {}, "chance": 0.5},
    "paired_associates": {"image_dir": "results/pam_final", "image_filter": {"prompt": "sandwich"},
                          "token_dir": TOKENS_PAM, "token_filter": {}, "chance": "1/n"},
}

ITEM_LENGTHS = [1, 8, 64, 256]   # L=1 added 2026-10-16: novel foils and serial order only
# Token items at this length cost about what a 256 px image does (see matched_figures). Images dashed, tokens solid,
# so the split does not rest on color alone.
MATCHED_L = 64
MATCHED_DATASETS = ["Brady2008", "things"]

TASK_TITLES = {"afc_recognition": "2AFC — Novel", "serial_order": "Serial AFC", "paired_associates": "PAM (cue → word)"}


def load_cells(run_dir, metrics=("correct",)):
    """One row per (model, task, condition minus n_trials), seeds pooled; Wilson CI for each 0/1 trial field."""
    groups = defaultdict(list)
    for path in sorted(Path(run_dir).glob("*/*.json")):
        d = json.loads(path.read_text())
        if d.get("error") or d.get("skipped"):
            continue
        cond = {k: v for k, v in d["run"]["condition"].items() if k != "n_trials"}
        groups[(d["run"]["model"]["name"], d["run"]["task"], json.dumps(cond, sort_keys=True))].append(d)
    rows = []
    for (model, task, cond_json), results in groups.items():
        trials = [t for r in results for t in r["trials"]]
        row = {"model": model, "task": task, "seeds": len(results), **json.loads(cond_json)}
        for field in metrics:
            # "correct" keeps bench.plot's column names (accuracy, ci95_low/high) so plot_series reads it as is.
            name, lo, hi = ("accuracy", "ci95_low", "ci95_high") if field == "correct" else \
                (field, f"{field}_ci95_low", f"{field}_ci95_high")
            row[name], row[lo], row[hi] = wilson_ci(sum(int(t[field] or 0) for t in trials), len(trials))
        rows.append(row)
    return rows


def compact(x):
    if x >= 10000:
        return f"{x / 1000:.0f}k"      # 27k, 122k (.2g would print 1.2e+02k)
    return f"{x / 1000:.2g}k" if x >= 1000 else f"{x:g}"


# Wider than bench.plot's 1.7: these grids mix sizes from several runs (927, 1,853, 2,500, 3,750) that crowd the top end.
LABEL_RATIO = 2.2


def set_axes(ax, sizes, decades=False, ratio=LABEL_RATIO):
    """bench.plot.apply_axes with short, thinned labels ("27k"). Small-multiple panels get decade ticks instead of
    one tick per study size, which crowd each other at panel width."""
    apply_axes(ax, sizes, max(sizes))
    if decades:
        ticks = [10 ** k for k in range(6) if 10 ** k <= max(sizes) * 1.3]
        ax.set_xticks(ticks)
        ax.set_xticklabels([compact(t) for t in ticks])
        return
    ticks = sorted(set(sizes))
    keep = [ticks[0]]
    for x in ticks[1:]:
        if x / keep[-1] >= ratio:
            keep.append(x)
    # The largest size is the one worth naming; drop the kept label just before it if the two would touch.
    if keep[-1] != ticks[-1]:
        if ticks[-1] / keep[-1] < ratio and len(keep) > 1:
            keep.pop()
        keep.append(ticks[-1])
    ax.set_xticks(ticks)
    ax.set_xticklabels([compact(x) if x in keep else "" for x in ticks])


def select(rows, **filt):
    return [r for r in rows if all(r.get(k) == v for k, v in filt.items())]


def chance_line(ax, y=50, label="Chance"):
    ax.axhline(y=y, color="gray", linestyle="--", alpha=0.5, linewidth=1, label=label, zorder=0)


def plot_chance(ax, chance, sizes):
    """A flat line for a fixed chance level, or the 1/n curve (as bench.plot.plot_chance) for paired associates."""
    if chance == "1/n":
        xs = sorted(set(sizes))
        ax.plot(xs, [100 / x for x in xs], color="gray", linestyle="--", alpha=0.5, linewidth=1,
                label="Chance (1/n)", zorder=0)
    elif chance:
        chance_line(ax, y=100 * chance)


def by_length_figure(rows, title, out_path, metric="accuracy", ylabel="Accuracy (%)", chance=0.5):
    """One column per item length, all model sizes on each axes; the layout of bench/plot_thinking.py."""
    lengths = [L for L in ITEM_LENGTHS if select(rows, item_length=L)]
    if not lengths:
        print(f"no results for {out_path.name}, skipped")
        return
    fig, axes = panel_grid(lengths, width=4.2)
    for ax, L in zip(axes, lengths):
        sel = select(rows, item_length=L)
        for model, label, color, marker in MODELS:
            cells = [r for r in sel if base_model(r["model"]) == model]
            if cells:
                plot_series(ax, cells, color, marker, "-", label, metric=metric)
        plot_chance(ax, chance, [r["n_images"] for r in sel])
        set_axes(ax, [r["n_images"] for r in sel], ratio=3.0)
        ax.set_ylabel(ylabel)
        ax.set_title(f"L = {L} tokens per item")
    finish_grid(fig, axes, title, out_path)


def token_figures(out_dir):
    full = load_cells(TOKENS_FULL)
    for task, filt, name, title in [
        ("afc_recognition", {"foil_type": "novel"}, "recognition_novel", "2AFC — Novel foil"),
        ("afc_recognition", {"foil_type": "recombined"}, "recognition_recombined",
         "2AFC — Recombined foil (halves of two studied items)"),
        ("serial_order", {}, "serial_order", "Serial AFC"),
    ]:
        by_length_figure(select(full, task=task, **filt), f"{title}, token items", out_dir / f"tokens_{name}.png")
    by_length_figure(load_cells(TOKENS_PAM), "Paired associates (token item → word), sandwich layout",
                     out_dir / "tokens_paired_associates.png", chance="1/n")

    recall = load_cells(RECALL, metrics=("correct", "first_token_top1", "exact_recall"))
    # Chance for either is ~1 / 245k (the token pool), so no chance line.
    by_length_figure(recall, "Cued recall: the first tail token is the top prediction",
                     out_dir / "tokens_recall_first_token.png", metric="first_token_top1",
                     ylabel="First-token top-1 (%)", chance=False)
    by_length_figure(recall, "Cued recall: the whole tail (L/2 tokens) reproduced greedily",
                     out_dir / "tokens_recall_exact.png", metric="exact_recall",
                     ylabel="Exact recall (%)", chance=False)

    recall_vs_recognition(full, recall, out_dir)
    ab_labeled_figure(out_dir)
    answer_format_figure(full, recall, out_dir)


def panel_grid(columns, width=4.6, rows=1):
    fig, axes = plt.subplots(rows, len(columns), figsize=(width * len(columns), 5.2 * rows), sharey=True,
                             squeeze=False)
    return fig, (axes[0] if rows == 1 else axes.ravel())


def finish_grid(fig, axes, title, out_path):
    for ax in axes:
        if not ax.get_subplotspec().is_first_col():
            ax.set_ylabel("")
    # One legend for the figure: every panel draws the same series in the same style.
    handles, labels = {}, []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in handles:
                handles[l] = h
                labels.append(l)
    fig.legend([handles[l] for l in labels], labels, loc="lower center", ncol=min(len(labels), 4),
               framealpha=0.9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(title, fontsize=16)
    # The legend strip is a fixed height, so its share of a taller (multi-row) figure is smaller.
    nrows = axes[0].get_subplotspec().get_gridspec().nrows
    fig.tight_layout(rect=(0, (0.12 if len(labels) > 4 else 0.08) / nrows, 1, 1 - 0.05 / nrows))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def recall_vs_recognition(full, recall, out_dir):
    """Same items, same study list: the explicit 1/2 judgment vs the likelihood of the true tail. Both chance 50%.
    One row per item length, one column per model size."""
    rec_all = select(full, task="afc_recognition", foil_type="novel")
    # L=1 has no recall probe (the cue would be half of one token), so it gets no row here.
    lengths = [L for L in ITEM_LENGTHS if L >= 2 and (select(recall, item_length=L) or select(rec_all, item_length=L))]
    models = [m for m in MODELS if any(r["model"] == m[0] for r in recall + rec_all)]
    fig, axes = panel_grid(models, width=4.2, rows=len(lengths))
    for i, L in enumerate(lengths):
        for j, (model, label, *_) in enumerate(models):
            ax = axes[i * len(models) + j]
            rec = select(rec_all, model=model, item_length=L)
            rcl = select(recall, model=model, item_length=L)
            if rec:
                plot_series(ax, rec, "#eb6834", "o", "-", "Recognition: 2AFC old vs new, answer 1/2")
            if rcl:
                plot_series(ax, rcl, "#2a78d6", "s", "--", "Cued recall: true vs unstudied tail, by likelihood")
            chance_line(ax)
            sizes = [r["n_images"] for r in rec + rcl]
            if sizes:
                set_axes(ax, sizes, decades=True)
            else:
                ax.set_xscale("log")
                ax.text(0.5, 0.5, "runs pending", transform=ax.transAxes, ha="center", va="center", color="gray")
            ax.set_ylabel(f"L = {L}\nAccuracy (%)")
            ax.set_title(label if i == 0 else "")
            if i < len(lengths) - 1:
                ax.set_xlabel("")
    finish_grid(fig, axes, "Recognition vs cued recall, token items", out_dir / "tokens_recall_vs_recognition.png")


def ab_labeled_figure(out_dir):
    rows = load_cells(AB_LABELED)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    for model, label, color, marker in MODELS:
        for prompt, desc, linestyle in [("original", "bare options", "-"), ("labeled", '"Item 1:" / "Item 2:"', "--")]:
            cells = select(rows, model=model, prompt=prompt)
            if cells:
                plot_series(ax, cells, color, marker, linestyle, f"{label} — {desc}")
    chance_line(ax)
    set_axes(ax, [r["n_images"] for r in rows])
    finish(ax, fig, "Option labels A/B (token items, L=8, novel foil)\nAccuracy vs Sequence Length",
           out_dir / "tokens_ab_labeled.png")


def answer_format_figure(full, recall, out_dir):
    """Same L=8 episodes, three ways of answering: the digit 1/2 (tokens_full), the studied item copied back and
    scored by likelihood (tokens_copy_probe), and cued recall of its tail (tokens_recall_probe). One column per
    model size. If copy tracks recall rather than the digit, the digit curves measure the answer step, not memory."""
    copy = load_cells(COPY_PROBE)
    if not copy:
        print("no copy-probe results yet; skipping tokens_answer_format.png")
        return
    L = 8
    digit = select(full, task="afc_recognition", foil_type="novel", prompt="original", item_length=L)
    models = [m for m in MODELS if any(r["model"] == m[0] for r in copy)]
    fig, axes = panel_grid(models, width=4.2)
    for ax, (model, label, *_) in zip(axes, models):
        series = [(select(digit, model=model), "#eb6834", "o", "-", "Digit: which option, answer 1/2"),
                  (select(copy, model=model, item_length=L), "#2a9d4f", "^", "-", "Copy: the studied option, by likelihood"),
                  (select(recall, model=model, item_length=L), "#2a78d6", "s", "--", "Cued recall: true vs unstudied tail")]
        for cells, color, marker, linestyle, desc in series:
            if cells:
                plot_series(ax, cells, color, marker, linestyle, desc)
        chance_line(ax)
        set_axes(ax, [r["n_images"] for cells, *_ in series for r in cells], decades=True)
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(label)
    finish_grid(fig, axes, f"Answer format, token items (L = {L}, novel foil)", out_dir / "tokens_answer_format.png")


def matched_figures(out_dir):
    """Token items at L=64 against one image dataset: one figure per (task, dataset), one column per model size,
    laid out as bench/plot_thinking.py's on-vs-off figures.

    The prompt cost per item matches: a 256 px image (Brady natively, THINGS resized) is 8 x 8 merged patches +
    2 boundary tokens = 66, a token item 64 IDs + the newline separator = 65. A model gets a column once its L=64
    token runs exist.
    """
    labels = {m: label for m, label, *_ in MODELS}
    for task, spec in MATCHED_TASKS.items():
        images = load_cells(spec["image_dir"])
        token_rows = select(load_cells(spec["token_dir"]), task=task, item_length=MATCHED_L, **spec["token_filter"])
        for dataset in MATCHED_DATASETS:
            image_rows = select(images, task=task, dataset=dataset, **spec["image_filter"])
            # Every model with image results gets a column, so the figure keeps its shape as token runs land.
            models = [m for m, *_ in MODELS if any(base_model(r["model"]) == m for r in image_rows + token_rows)]
            fig, axes = panel_grid(models, width=4.2)
            for ax, model in zip(axes, models):
                conditions = [
                    ([r for r in image_rows if base_model(r["model"]) == model],
                     f"{DATASET_LABELS[dataset]} images, 256 px (66 tokens)", "#2a78d6", "o", "-"),
                    ([r for r in token_rows if r["model"] == model],
                     f"Token items, L={MATCHED_L} (65 tokens)", "#eb6834", "s", "--"),
                ]
                for cells, desc, color, marker, linestyle in conditions:
                    if cells:
                        plot_series(ax, cells, color, marker, linestyle, desc)
                sizes = [r["n_images"] for cells, *_ in conditions for r in cells]
                plot_chance(ax, spec["chance"], sizes)
                # Panels are half a single figure's width, so labels need more room.
                set_axes(ax, sizes, ratio=3.0)
                pending = not conditions[1][0]
                ax.set_title(labels[model] + ("\n(token runs pending)" if pending else ""),
                             fontsize=13 if pending else None)
            finish_grid(fig, axes, f"{TASK_TITLES[task]} ({DATASET_LABELS[dataset]}): images vs token items, "
                        f"~66 tokens per item", out_dir / f"tokens_vs_images_{task}_{dataset}.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="plots")
    args = parser.parse_args()
    out_dir = Path(args.out) / "tokens"
    out_dir.mkdir(parents=True, exist_ok=True)
    token_figures(out_dir)
    matched_figures(out_dir)


if __name__ == "__main__":
    main()
