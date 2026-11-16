"""Config-driven patching sweep (INTERP_PLAN.md, sections 3 and 6).

  python -m interp.phase2_3.run_patch configs/interp/<phase>/<config>.yaml [--dry-run] [--overwrite]

One result file per (task, dataset, n, seed), under <output_dir>/<run_name>/<task>/, with every trial's logit
differences for every (direction, surface, unit), the config and provenance embedded, as the benchmark does.
Existing files are skipped so a job resumes.

Per trial: the clean and corrupted prompts are rendered, each prefix is run once for its cache, and each
direction then copies pieces of the source run into a fork of the base run before the probe:
  clean_to_corrupt   base = corrupted cache, source = clean   ("is this piece sufficient to bring the answer back")
  corrupt_to_clean   base = clean cache, source = corrupted   ("is this piece necessary to keep the clean answer")
Surfaces: all (M + conv + K/V, the ceiling), all_M, all_conv, all_KV, M_by_layer, KV_by_layer,
heads_by_layer (o_proj input at every probe position, one head at a time), residual_at_probe (decoder layer
output at every probe position, one layer at a time), residual_at_study_span (below).
Recovery for a patch = (LD_patched - LD_base) / (LD_source - LD_base).

residual_at_study_span re-runs the base prefix with one decoder layer's output overwritten by the source's over a
span of prefix positions, then runs the probe from the resulting cache. Spans: "item", the positions whose input
embeddings differ between the two prompts (the manipulated item; both swapped slots for serial order, both copies
of the word for PAM), and "after", every prefix position after the last of those. "item/emb" overwrites the
item's input embeddings instead, which makes the re-run the source run, so its recovery should be 1 (a check).
Each unit is probed three ways: as is, with every DeltaNet state (M and conv) reset to the unpatched base
run's ("/reset_M"), and with every full-attention layer's keys/values reset ("/reset_KV"), to say which channel
carries a study-time patch to the answer.
"""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import torch
import yaml

from bench.config import provenance
from bench.models import build_evaluator
from interp import pairs
from interp.harness import CachePatch, Capture, Harness, Override, run_patched

SURFACES = ("all", "all_M", "all_conv", "all_KV", "M_by_layer", "KV_by_layer", "heads_by_layer", "residual_at_probe",
            "residual_at_study_span")
DIRECTIONS = {"clean_to_corrupt": ("corrupt", "clean"), "corrupt_to_clean": ("clean", "corrupt")}


def load_config(path):
    cfg = yaml.safe_load(Path(path).read_text())
    for key in ("run_name", "model", "tasks", "seeds"):
        if not cfg.get(key):
            raise ValueError(f"config is missing {key}")
    cfg.setdefault("output_dir", "results/interp")
    cfg.setdefault("surfaces", list(SURFACES))
    cfg.setdefault("directions", list(DIRECTIONS))
    unknown = set(cfg["surfaces"]) - set(SURFACES)
    if unknown:
        raise ValueError(f"unknown surfaces {sorted(unknown)}; expected {SURFACES}")
    unknown = set(cfg["directions"]) - set(DIRECTIONS)
    if unknown:
        raise ValueError(f"unknown directions {sorted(unknown)}; expected {sorted(DIRECTIONS)}")
    return cfg


def expand_cells(cfg):
    cells = []
    for task_cfg in cfg["tasks"]:
        for n in task_cfg["n_images"]:
            for seed in cfg["seeds"]:
                cond = {k: v for k, v in task_cfg.items() if k not in ("task", "n_images")}
                cond["n_images"] = n
                cells.append({"task": task_cfg["task"], "condition": cond, "seed": seed})
    return cells


def cell_path(cfg, cell):
    parts = "_".join(f"{k}-{v}" for k, v in sorted(cell["condition"].items()))
    return Path(cfg["output_dir"]) / cfg["run_name"] / cell["task"] / f"{cfg['model']['name']}__{parts}__seed{cell['seed']}.json"


def is_done(path):
    """A cell counts as done only if its file exists and holds no error (a failed cell is retried, as the bench does)."""
    if not path.exists():
        return False
    try:
        return json.loads(path.read_text()).get("error") is None
    except json.JSONDecodeError:
        return False


def pair_kwargs(cond):
    return {k: cond[k] for k in ("item_length", "token_pool", "wordpool") if k in cond}


def patch_units(h, surface, source, direction_captures):
    """(unit name, cache patches, hooks) for every unit of a surface."""
    if surface == "all":
        return [("all", [CachePatch("M", h.linear_layers, source), CachePatch("conv", h.linear_layers, source),
                         CachePatch("KV", h.full_layers, source)], [])]
    if surface == "all_M":
        return [("all", [CachePatch("M", h.linear_layers, source)], [])]
    if surface == "all_conv":
        return [("all", [CachePatch("conv", h.linear_layers, source)], [])]
    if surface == "all_KV":
        return [("all", [CachePatch("KV", h.full_layers, source)], [])]
    if surface == "M_by_layer":
        return [(f"L{i}", [CachePatch("M", [i], source)], []) for i in h.linear_layers]
    if surface == "KV_by_layer":
        return [(f"L{i}", [CachePatch("KV", [i], source)], []) for i in h.full_layers]
    if surface == "residual_at_probe":
        return [(f"L{i}", [], [Override("residual", i, direction_captures["residual"][i])]) for i in range(len(h.layers))]
    if surface == "heads_by_layer":
        n_heads = h.inner.language_model.config.num_attention_heads
        return [(f"L{i}H{hd}", [], [Override("heads", i, direction_captures["heads"][i], heads=[hd])])
                for i in h.full_layers for hd in range(n_heads)]
    raise ValueError(surface)


def capture_source(h, prompt, cache, need_residual, need_heads):
    """One probe run on the source cache that records what the hook surfaces will inject."""
    caps = []
    if need_residual:
        caps += [Capture("residual", i) for i in range(len(h.layers))]
    if need_heads:
        caps += [Capture("heads", i) for i in h.full_layers]
    if caps:
        h.run_probe(prompt, cache, hooks=caps)
    out = {"residual": {}, "heads": {}}
    for c in caps:
        out[c.surface][c.layer] = c.value
    return out


def study_spans(emb_a, emb_b, prefix_len):
    """Prefix positions whose input embeddings differ (the manipulated item or items), and every prefix position after."""
    item = (emb_a != emb_b).any(-1)[0].nonzero().flatten().tolist()
    if not item:
        raise ValueError("clean and corrupted prefixes have identical input embeddings")
    return {"item": item, "after": list(range(item[-1] + 1, prefix_len))}


def contiguous_runs(positions):
    return 1 + sum(b != a + 1 for a, b in zip(positions, positions[1:]))


def prefix_captures(h):
    return [Capture("embeds", 0)] + [Capture("residual", i) for i in range(len(h.layers))]


def study_span_patches(h, prompt, base_cache, source_caps, spans):
    """{unit: probe logits} for the study-span surface: the base prefix re-run with each override, probed three ways."""
    embeds, residual = source_caps[0].value, {c.layer: c.value for c in source_caps[1:]}
    units = [("item/emb", Override("embeds", 0, embeds, positions=spans["item"]))]
    for span, positions in spans.items():
        if positions:
            units += [(f"{span}/L{i}", Override("residual", i, residual[i], positions=positions)) for i in range(len(h.layers))]
    resets = {"": [],
              "/reset_M": [CachePatch("M", h.linear_layers, base_cache), CachePatch("conv", h.linear_layers, base_cache)],
              "/reset_KV": [CachePatch("KV", h.full_layers, base_cache)]}
    out = {}
    for unit, hook in units:
        patched_cache = h.run_prefix(prompt, hooks=[hook])
        for suffix, patches in resets.items():
            out[unit + suffix] = run_patched(h, prompt, patched_cache, patches)
        del patched_cache
    return out


def run_trial(h, pair, cfg):
    prompts = {"clean": h.prepare(pair.clean, prefill=pair.prefill), "corrupt": h.prepare(pair.corrupt, prefill=pair.prefill)}
    if prompts["clean"].length != prompts["corrupt"].length or prompts["clean"].split != prompts["corrupt"].split:
        raise ValueError("clean and corrupted prompts are not aligned")
    study_span = "residual_at_study_span" in cfg["surfaces"]
    # The study-span surface needs every layer's output over the prefix of both runs; record it on the prefix runs
    # that make the caches, rather than running each prefix twice.
    prefix_caps = {k: prefix_captures(h) if study_span else [] for k in prompts}
    caches = {k: h.run_prefix(p, hooks=prefix_caps[k]) for k, p in prompts.items()}
    spans = study_spans(prefix_caps["clean"][0].value, prefix_caps["corrupt"][0].value, prompts["clean"].split) if study_span else None
    ld = {k: h.logit_diff(h.run_probe(prompts[k], caches[k]), pair.answer, pair.counterfactual) for k in prompts}
    need_residual = "residual_at_probe" in cfg["surfaces"]
    need_heads = "heads_by_layer" in cfg["surfaces"]

    patched = {}
    for direction in cfg["directions"]:
        base, source = DIRECTIONS[direction]
        captures = capture_source(h, prompts[source], caches[source], need_residual, need_heads)
        patched[direction] = {}
        for surface in cfg["surfaces"]:
            patched[direction][surface] = {}
            if surface == "residual_at_study_span":
                logits = study_span_patches(h, prompts[base], caches[base], prefix_caps[source], spans)
                patched[direction][surface] = {u: h.logit_diff(x, pair.answer, pair.counterfactual) for u, x in logits.items()}
                continue
            for unit, cache_patches, hooks in patch_units(h, surface, caches[source], captures):
                # Hook surfaces inject the source's probe-position activations into the base run; cache surfaces
                # edit a fork of the base cache. The probe prompt is the same tokens either way.
                logits = run_patched(h, prompts[base], caches[base], cache_patches, hooks=hooks)
                patched[direction][surface][unit] = h.logit_diff(logits, pair.answer, pair.counterfactual)
        del captures
    del caches, prefix_caps
    return {
        "answer": pair.answer, "counterfactual": pair.counterfactual, "slot": pair.slot, "meta": pair.meta,
        "prompt_tokens": prompts["clean"].length, "probe_tokens": prompts["clean"].length - prompts["clean"].split,
        **({"span": {"item_tokens": len(spans["item"]), "item_runs": contiguous_runs(spans["item"]),
                     "item_start": spans["item"][0], "after_tokens": len(spans["after"])}} if spans else {}),
        "ld_clean": ld["clean"], "ld_corrupt": ld["corrupt"], "patched": patched,
    }


def summarize(trials, cfg):
    """Mean recovery and the fraction of trials whose answer sign is restored, per direction / surface / unit."""
    out = {"n_trials": len(trials),
           "mean_ld_clean": sum(t["ld_clean"] for t in trials) / len(trials),
           "mean_ld_corrupt": sum(t["ld_corrupt"] for t in trials) / len(trials),
           "frac_clean_correct": sum(t["ld_clean"] > 0 for t in trials) / len(trials),
           "frac_corrupt_correct": sum(t["ld_corrupt"] < 0 for t in trials) / len(trials),
           "recovery": {}, "frac_restored": {}}
    for direction in cfg["directions"]:
        base_key, source_key = DIRECTIONS[direction]
        out["recovery"][direction], out["frac_restored"][direction] = {}, {}
        for surface in cfg["surfaces"]:
            rec, restored = {}, {}
            for unit in trials[0]["patched"][direction][surface]:
                vals, signs = [], []
                for t in trials:
                    base, source = t[f"ld_{base_key}"], t[f"ld_{source_key}"]
                    x = t["patched"][direction][surface][unit]
                    if abs(source - base) > 1e-6:
                        vals.append((x - base) / (source - base))
                    signs.append((x > 0) == (source > 0))
                rec[unit] = sum(vals) / len(vals) if vals else None
                restored[unit] = sum(signs) / len(signs)
            out["recovery"][direction][surface] = rec
            out["frac_restored"][direction][surface] = restored
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    cells = expand_cells(cfg)
    pending = [c for c in cells if args.overwrite or not is_done(cell_path(cfg, c))]
    print(f"{len(cells)} cells, {len(pending)} to do; surfaces {cfg['surfaces']}; directions {cfg['directions']}", flush=True)
    if args.dry_run:
        for c in cells:
            print(f"  [{'todo' if c in pending else 'done'}] {cell_path(cfg, c)}")
        return
        return
    if not pending:
        return

    ev = build_evaluator({"device_map": "cuda", **cfg["model"]})
    h = Harness(ev)
    prov = provenance(cfg)
    for k, cell in enumerate(pending, 1):
        cond, task, seed = cell["condition"], cell["task"], cell["seed"]
        print(f"[{k}/{len(pending)}] {task} {cond} seed={seed}", flush=True)
        t0, trials, error = time.time(), [], None
        torch.cuda.reset_peak_memory_stats()
        try:
            for i in range(cond["n_trials"]):
                pair = pairs.make_pair(ev, task, cond["dataset"], cond["n_images"], seed=seed, trial=i, **pair_kwargs(cond))
                trials.append({"trial": i, **run_trial(h, pair, cfg)})
                if i == 0:
                    print(f"  trial 0: {trials[0]['prompt_tokens']} tokens, {time.time() - t0:.1f}s", flush=True)
        except Exception:
            error = traceback.format_exc()
            print(f"  FAILED:\n{error}", flush=True)
            torch.cuda.empty_cache()
        result = {"run": {"model": cfg["model"], "task": task, "condition": cond, "seed": seed}, "config": cfg,
                  "layers": {"linear": h.linear_layers, "full": h.full_layers},
                  "provenance": prov, "elapsed_s": round(time.time() - t0, 1),
                  "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2), "error": error,
                  "trials": trials, "metrics": summarize(trials, cfg) if trials and not error else {}}
        out = cell_path(cfg, cell)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=1))
        if result["metrics"]:
            m = result["metrics"]
            heads = {d: {s: v.get("all") for s, v in m["recovery"][d].items() if "all" in v} for d in m["recovery"]}
            print(f"  clean LD {m['mean_ld_clean']:+.2f} ({m['frac_clean_correct']:.2f} correct), corrupt LD {m['mean_ld_corrupt']:+.2f} "
                  f"({m['frac_corrupt_correct']:.2f} correct); recovery {heads} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
