#!/usr/bin/env python
"""Figures for a patching run (interp/phase2_3/run_patch.py results).

  python -m interp.phase2_3.plot_patch results/interp/phase2_brady_4b [--out plots/interp/phase2_brady_4b] [--metric flip|ld]

One figure per view, each a grid of task (rows) x list size (columns):
  surfaces.png   recovery for the coarse surfaces (all, M, conv, K/V), both directions
  layers.png     recovery by layer for K/V (full-attention layers), M (DeltaNet layers) and the residual at the probe
  heads.png      recovery by (full-attention layer, head)
  study_span.png        recovery by layer for the residual patched over the study-list item, over every prefix
                        position after it, and at the probe (residual_at_study_span runs)
  study_span_split.png  the two study-span curves again, each also with the DeltaNet states or the keys/values
                        reset to the unpatched run before the probe: which channel carries the patch to the answer
A view is drawn only if the run has its surfaces.
layers.png and heads.png are the clean -> corrupted direction; layers_reverse.png and heads_reverse.png the other one.
Recovery is (LD_patched - LD_base) / (LD_source - LD_base), averaged over trials; 1 means the swapped piece carries
the whole answer, 0 none of it. With --metric flip the same figures show instead the fraction of trials whose answer
the patch flipped to the source run's (surfaces_flip.png, layers_flip.png, heads_flip.png). The flip fraction misses
pieces that move the answer only part of the way, which is most single layers and heads, so recovery stays the default.
With --metric ld they show the raw logit difference (score of the clean answer minus the corrupted one) after the
patch, averaged over trials, with dashed lines at the unpatched clean and corrupted means (surfaces_ld.png, ...).
Panels then keep their own y-scales, since the gap between the two runs differs by task.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TASK_LABELS = {"afc_recognition": "Recognition", "serial_order": "Serial order", "paired_associates": "Paired associates"}
SURFACES = [("all", "all (ceiling)"), ("all_M", "all M"), ("all_conv", "all conv"), ("all_KV", "all K/V")]
DIRECTION_SUFFIX = {"clean_to_corrupt": "", "corrupt_to_clean": "_reverse"}
DIRECTION_STYLE = {"clean_to_corrupt": ("#2a78d6", "clean → corrupted"), "corrupt_to_clean": ("#c96a1e", "corrupted → clean")}
# metric name -> (key in a cell's metrics, axis label, title word, file suffix)
METRICS = {"recovery": ("recovery", "recovery", "Recovery", ""),
           "flip": ("frac_restored", "fraction of trials flipped", "Answer flips", "_flip"),
           "ld": ("mean_ld", "logit difference\n(clean answer − corrupted)", "Logit difference", "_ld")}
REF_STYLE = {"mean_ld_clean": ("#2a78d6", "unpatched clean"), "mean_ld_corrupt": ("#c96a1e", "unpatched corrupted")}
LAYER_SERIES = [("KV_by_layer", "K/V (full attention)", "#2a78d6", "o"),
                ("M_by_layer", "M (DeltaNet)", "#1b9e77", "s"),
                ("residual_at_probe", "residual at probe", "#8c8c8c", "^")]


SPAN_STYLE = {"item": ("#2a78d6", "o", "item span"), "after": ("#c96a1e", "s", "everything after the item")}
RESET_STYLE = {"": ("-", "as is"), "/reset_M": ("--", "DeltaNet M + conv reset"), "/reset_KV": (":", "K/V reset")}


def load(run_dir):
    cells = {}
    for path in sorted(Path(run_dir).glob("*/*.json")):
        r = json.loads(path.read_text())
        if r.get("error") or not r.get("metrics"):
            continue
        r["metrics"]["mean_ld"] = mean_patched_ld(r["trials"])
        cells[(r["run"]["task"], r["run"]["condition"]["n_images"])] = r
    return cells


def mean_patched_ld(trials):
    """{direction: {surface: {unit: mean patched LD over trials}}}, the raw-LD counterpart of the recovery dict."""
    out = {}
    for d, by_surface in trials[0]["patched"].items():
        out[d] = {s: {u: float(np.mean([t["patched"][d][s][u] for t in trials])) for u in units}
                  for s, units in by_surface.items()}
    return out


def reference_lines(ax, m):
    """Unpatched clean and corrupted mean LDs, the two ends a patch moves between (raw-LD figures only)."""
    for key, (color, label) in REF_STYLE.items():
        ax.axhline(m[key], color=color, linewidth=0.9, linestyle="--", label=label)


def grid(cells):
    """Rows are tasks; column j of a row is that task's j-th list size (tasks have their own size grids)."""
    tasks = [t for t in TASK_LABELS if any(k[0] == t for k in cells)]
    sizes = {t: sorted(k[1] for k in cells if k[0] == t) for t in tasks}
    return tasks, sizes


def panels(cells):
    tasks, sizes = grid(cells)
    ncol = max(len(v) for v in sizes.values())
    return tasks, sizes, ncol


def legend_from(axes):
    for row in axes:
        for ax in row:
            h, l = ax.get_legend_handles_labels()
            if h:
                return h, l
    return [], []


def style(ax):
    ax.grid(True, color="#e6e6e6", linewidth=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def plot_surfaces(cells, out, metric="recovery"):
    key, ylabel, word, suffix = METRICS[metric]
    tasks, sizes, ncol = panels(cells)
    fig, axes = plt.subplots(len(tasks), ncol, figsize=(5.2 * ncol, 2.6 * len(tasks)), squeeze=False, sharey=metric != "ld")
    for i, task in enumerate(tasks):
        for j in range(ncol):
            ax = axes[i][j]
            if j >= len(sizes[task]):
                ax.set_visible(False)
                continue
            n = sizes[task][j]
            r = cells[(task, n)]
            m = r["metrics"]
            x = np.arange(len(SURFACES))
            for k, (d, (color, label)) in enumerate(DIRECTION_STYLE.items()):
                if d not in m[key]:
                    continue
                vals = [m[key][d][s]["all"] for s, _ in SURFACES]
                ax.bar(x + (k - 0.5) * 0.36, vals, width=0.34, color=color, label=label, linewidth=0)
            ax.axhline(0, color="#888888", linewidth=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels([l for _, l in SURFACES], fontsize=8, rotation=20, ha="right")
            if metric == "ld":
                reference_lines(ax, m)
            else:
                ax.set_ylim(-0.2, 1.15)
            style(ax)
            ax.set_title(f"{TASK_LABELS[task]}, n = {n}", fontsize=9)
            if j == 0:
                ax.set_ylabel(ylabel, fontsize=9)
            if metric != "ld":   # the raw-LD panels draw these as reference lines instead
                ax.text(0.02, 0.95, f"LD clean {m['mean_ld_clean']:+.1f}, corrupted {m['mean_ld_corrupt']:+.1f}", transform=ax.transAxes,
                        fontsize=7, va="top", color="#555555")
    handles, labels = legend_from(axes)
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"{word} from swapping whole cache surfaces (50 trials per cell)", fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(out / f"surfaces{suffix}.png", dpi=160)
    plt.close(fig)


def plot_layers(cells, out, direction="clean_to_corrupt", metric="recovery"):
    key, ylabel, word, suffix = METRICS[metric]
    tasks, sizes, ncol = panels(cells)
    fig, axes = plt.subplots(len(tasks), ncol, figsize=(5.8 * ncol, 2.6 * len(tasks)), squeeze=False, sharey=metric != "ld")
    for i, task in enumerate(tasks):
        for j in range(ncol):
            ax = axes[i][j]
            if j >= len(sizes[task]):
                ax.set_visible(False)
                continue
            n = sizes[task][j]
            rec = cells[(task, n)]["metrics"][key][direction]
            for surface, label, color, marker in LAYER_SERIES:
                if surface not in rec:
                    continue
                pts = sorted((int(u[1:]), v) for u, v in rec[surface].items() if v is not None)
                ax.plot([p[0] for p in pts], [p[1] for p in pts], marker=marker, markersize=3.5, linewidth=1.2, color=color, label=label)
            ax.axhline(0, color="#888888", linewidth=0.8)
            if metric == "ld":
                reference_lines(ax, cells[(task, n)]["metrics"])
            else:
                ax.set_ylim(-0.2, 1.15)
            style(ax)
            ax.set_title(f"{TASK_LABELS[task]}, n = {n}", fontsize=9)
            if j == 0:
                ax.set_ylabel(ylabel, fontsize=9)
            if i == len(tasks) - 1:
                ax.set_xlabel("layer", fontsize=9)
    handles, labels = legend_from(axes)
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"{word} by layer, {DIRECTION_STYLE[direction][1]}", fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(out / f"layers{DIRECTION_SUFFIX[direction]}{suffix}.png", dpi=160)
    plt.close(fig)


def plot_heads(cells, out, direction="clean_to_corrupt", metric="recovery"):
    key, ylabel, word, suffix = METRICS[metric]
    tasks, sizes, ncol = panels(cells)
    fig, axes = plt.subplots(len(tasks), ncol, figsize=(5.4 * ncol, 2.8 * len(tasks)), squeeze=False)
    vmax = 0
    mats = {}
    for task in tasks:
        for n in sizes[task]:
            r = cells[(task, n)]
            if "heads_by_layer" not in r["metrics"][key][direction]:
                continue
            rec = r["metrics"][key][direction]["heads_by_layer"]
            layers = sorted({int(u.split("H")[0][1:]) for u in rec})
            heads = sorted({int(u.split("H")[1]) for u in rec})
            mat = np.full((len(layers), len(heads)), np.nan)
            for u, v in rec.items():
                l, h = u.split("H")
                mat[layers.index(int(l[1:])), heads.index(int(h))] = v
            mats[(task, n)] = (layers, heads, mat)
            vmax = max(vmax, np.nanmax(mat))
    if not mats:
        plt.close(fig)
        return   # run without the per-head sweep
    for i, task in enumerate(tasks):
        for j in range(ncol):
            ax = axes[i][j]
            if j >= len(sizes[task]) or (task, sizes[task][j]) not in mats:
                ax.set_visible(False)
                continue
            n = sizes[task][j]
            layers, heads, mat = mats[(task, n)]
            if metric == "ld":
                # Each panel spans its own unpatched corrupted (red) to clean (blue) LD; the marked head is the one that
                # moves the base run furthest toward the source run.
                m = cells[(task, n)]["metrics"]
                lo, hi = m["mean_ld_corrupt"], m["mean_ld_clean"]
                base, source = (lo, hi) if direction == "clean_to_corrupt" else (hi, lo)
                im = ax.imshow(mat, cmap="RdBu", vmin=lo, vmax=hi, aspect="auto")
                fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02).ax.tick_params(labelsize=6)
                top = np.unravel_index(np.nanargmax((mat - base) * np.sign(source - base)), mat.shape)
                ax.text(top[1], top[0], f"{mat[top]:+.1f}", ha="center", va="center", fontsize=6, color="black")
            else:
                im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=vmax, aspect="auto")
                top = np.unravel_index(np.nanargmax(mat), mat.shape)
                ax.text(top[1], top[0], f"{mat[top]:.2f}", ha="center", va="center", fontsize=6, color="white" if mat[top] > vmax / 2 else "black")
            ax.set_yticks(range(len(layers)))
            ax.set_yticklabels([f"L{l}" for l in layers], fontsize=7)
            ax.set_xticks(range(0, len(heads), 2))
            ax.set_xticklabels([str(h) for h in heads[::2]], fontsize=7)
            ax.set_title(f"{TASK_LABELS[task]}, n = {n}", fontsize=9)
            if i == len(tasks) - 1:
                ax.set_xlabel("head", fontsize=9)
    if metric == "ld":
        note = "colour runs from the unpatched corrupted LD (red) to the clean LD (blue); number marks the head that moves it furthest toward the other run"
    else:
        fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label=ylabel)
        note = "number marks the top head"
    fig.suptitle(f"{word} from one head's output at the probe, {DIRECTION_STYLE[direction][1]}\n({note})", fontsize=10)
    fig.savefig(out / f"heads{DIRECTION_SUFFIX[direction]}{suffix}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def span_curve(rec, span, suffix=""):
    """[(layer, value)] for one study span and reset variant, from units like item/L5 or item/L5/reset_M."""
    pts = []
    for u, v in rec.items():
        parts = u.split("/")
        if parts[0] == span and parts[1].startswith("L") and "/".join(parts[2:]) == suffix.lstrip("/") and v is not None:
            pts.append((int(parts[1][1:]), v))
    return sorted(pts)


def mark_full_layers(ax, cell):
    for layer in cell.get("layers", {}).get("full", []):
        ax.axvline(layer, color="#d8d8d8", linewidth=0.8, zorder=0)


def plot_study_span(cells, out, direction="clean_to_corrupt", metric="recovery", split=False):
    key, ylabel, word, suffix = METRICS[metric]
    tasks, sizes, ncol = panels(cells)
    fig, axes = plt.subplots(len(tasks), ncol, figsize=(5.8 * ncol, 2.8 * len(tasks)), squeeze=False, sharey=metric != "ld")
    for i, task in enumerate(tasks):
        for j in range(ncol):
            ax = axes[i][j]
            if j >= len(sizes[task]):
                ax.set_visible(False)
                continue
            n = sizes[task][j]
            cell = cells[(task, n)]
            rec = cell["metrics"][key][direction]
            mark_full_layers(ax, cell)
            span_rec = rec.get("residual_at_study_span", {})
            for span, (color, marker, label) in SPAN_STYLE.items():
                for reset, (ls, reset_label) in (RESET_STYLE.items() if split else [("", RESET_STYLE[""])]):
                    pts = span_curve(span_rec, span, reset)
                    if pts:
                        ax.plot([p[0] for p in pts], [p[1] for p in pts], linestyle=ls, marker=marker if not reset else None,
                                markersize=3, linewidth=1.2, color=color, label=f"{label}, {reset_label}" if split else f"residual at {label}")
            if not split and "residual_at_probe" in rec:
                pts = sorted((int(u[1:]), v) for u, v in rec["residual_at_probe"].items() if v is not None)
                ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="^", markersize=3, linewidth=1.2, color="#8c8c8c",
                        label="residual at probe")
            if span_rec.get("item/emb") is not None:
                ax.plot([-1], [span_rec["item/emb"]], marker="D", markersize=4, color="#2a78d6", linestyle="none",
                        label="item embeddings (check: should be the source run)")
            ax.axhline(0, color="#888888", linewidth=0.8)
            if metric == "ld":
                reference_lines(ax, cell["metrics"])
            else:
                ax.set_ylim(-0.2, 1.15)
            style(ax)
            ax.set_title(f"{TASK_LABELS[task]}, n = {n}", fontsize=9)
            if j == 0:
                ax.set_ylabel(ylabel, fontsize=9)
            if i == len(tasks) - 1:
                ax.set_xlabel("layer (−1 = input embeddings; grey lines = full-attention layers)", fontsize=8)
    handles, labels = legend_from(axes)
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 4), fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.01))
    what = "with each channel reset before the probe" if split else "residual patched during study vs at the probe"
    fig.suptitle(f"{word} by layer, {what}, {DIRECTION_STYLE[direction][1]}", fontsize=10)
    fig.tight_layout(rect=(0, 0.08 if split else 0.05, 1, 0.96))
    fig.savefig(out / f"study_span{'_split' if split else ''}{DIRECTION_SUFFIX[direction]}{suffix}.png", dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--metric", choices=sorted(METRICS), default="recovery")
    args = ap.parse_args()
    out = Path(args.out or f"plots/interp/{Path(args.run_dir).name}")
    out.mkdir(parents=True, exist_ok=True)
    cells = load(args.run_dir)
    if not cells:
        raise SystemExit(f"no finished cells under {args.run_dir}")
    surfaces = set(next(iter(cells.values()))["config"]["surfaces"])
    views = []
    if {s for s, _ in SURFACES} <= surfaces:
        plot_surfaces(cells, out, metric=args.metric)
        views.append("surfaces")
    directions = [d for d in DIRECTION_STYLE if any(d in c["metrics"]["recovery"] for c in cells.values())]
    for d in directions:
        if surfaces & {"KV_by_layer", "M_by_layer"}:
            plot_layers(cells, out, direction=d, metric=args.metric)
        plot_heads(cells, out, direction=d, metric=args.metric)
        if "residual_at_study_span" in surfaces:
            plot_study_span(cells, out, direction=d, metric=args.metric)
            plot_study_span(cells, out, direction=d, metric=args.metric, split=True)
    views += ["layers", "heads"] if surfaces & {"KV_by_layer", "M_by_layer"} else []
    views += ["study_span", "study_span_split"] if "residual_at_study_span" in surfaces else []
    print(f"wrote {out}: {', '.join(views)} for {', '.join(directions)} ({args.metric}) from {len(cells)} cells")


if __name__ == "__main__":
    main()
