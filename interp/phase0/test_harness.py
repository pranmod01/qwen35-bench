"""Phase 0 substrate checks (INTERP_PLAN.md, section 7), one GPU job:

  python -m interp.phase0.test_harness --model Qwen/Qwen3.5-4B --n 25 --trials 3 [--datasets Brady2008 tokens]

Each check prints PASS/FAIL with the numbers it saw; the exit code is the number of failures. Checks 1-7:
 1 prefix + probe from the cache reproduces the single-pass answer logits
 2 a fork can be run twice with identical logits and leaves the golden cache untouched
 3 cache shapes: M [1, H, 128, 128] at every linear layer, keys/values [1, kv_heads, T, 256] at every full layer
 4 self-patch is a no-op
 5 all-swap ceiling: every M, conv and K/V from clean turns the corrupted answer into the clean one
 6 residual override at the last layer reproduces clean logits; a head override runs and changes the logits
 7 a backward pass through the DeltaNet kernel yields finite gradients at a middle layer's residual
 9 study span: the positions whose input embeddings differ are one contiguous run (the recognition item), and
   re-running the corrupted prefix with those embeddings overwritten by the clean ones reproduces the clean probe
   (check 8 is nnsight, in test_nnsight.py)
Also printed, as a preview of Phase 2: clean vs corrupted logit difference and the recovery from all-M / all-KV.
"""
import argparse
import json
import time

import torch
from transformers.cache_utils import LinearAttentionCacheLayerMixin

from bench.models import build_evaluator
from interp import pairs
from interp.harness import Capture, CachePatch, Harness, Override, fork_cache, run_patched
from interp.phase2_3.run_patch import contiguous_runs, study_spans

LD_TOL = 0.5         # logit-difference tolerance between split and single-pass runs: absolute, or ...
LD_REL_TOL = 0.05    # ... this fraction of the clean-to-corrupted gap, whichever is larger (bf16 chunk-boundary noise)
EXACT_TOL = 1e-3     # for runs that should be numerically identical


class Report:
    def __init__(self):
        self.rows = []

    def check(self, name, ok, detail):
        self.rows.append({"check": name, "pass": bool(ok), "detail": detail})
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)

    @property
    def failures(self):
        return sum(not r["pass"] for r in self.rows)


def ld(h, logits, pair):
    return h.logit_diff(logits, pair.answer, pair.counterfactual)


def cache_signature(cache):
    """Sums of M and lengths of K/V, to detect any mutation of a golden cache."""
    sig = []
    for layer in cache.layers:
        if isinstance(layer, LinearAttentionCacheLayerMixin):
            sig.append(("M", float(layer.recurrent_states[0].double().sum()), float(layer.conv_states[0].double().sum())))
        else:
            sig.append(("KV", int(layer.keys.shape[-2]), float(layer.keys.double().sum())))
    return sig


def run_checks(h, pair, report, tag):
    clean, corrupt = h.prepare(pair.clean), h.prepare(pair.corrupt)
    assert clean.split == corrupt.split and clean.length == corrupt.length, (clean.split, corrupt.split, clean.length, corrupt.length)
    assert torch.equal(clean.inputs["input_ids"][:, clean.split:], corrupt.inputs["input_ids"][:, corrupt.split:]), "probes differ"
    print(f"  {tag}: {clean.length} tokens, probe starts at {clean.split} ({clean.length - clean.split} probe tokens), "
          f"slot {pair.slot} of {pair.n}, answer {pair.answer}")

    # 1: split vs single pass
    full_c, full_x = h.full_pass(clean), h.full_pass(corrupt)
    cache_c, cache_x = h.run_prefix(clean), h.run_prefix(corrupt)
    split_c, split_x = h.run_probe(clean, cache_c), h.run_probe(corrupt, cache_x)
    ld_full_c, ld_split_c = ld(h, full_c, pair), ld(h, split_c, pair)
    ld_full_x, ld_split_x = ld(h, full_x, pair), ld(h, split_x, pair)
    top_same = int(full_c.argmax()) == int(split_c.argmax()) and int(full_x.argmax()) == int(split_x.argmax())
    max_abs = max(float((full_c - split_c).abs().max()), float((full_x - split_x).abs().max()))
    tol = max(LD_TOL, LD_REL_TOL * abs(ld_full_c - ld_full_x))
    report.check(f"1 split==single-pass [{tag}]",
                 abs(ld_full_c - ld_split_c) <= tol and abs(ld_full_x - ld_split_x) <= tol and top_same,
                 f"LD clean full {ld_full_c:+.3f} split {ld_split_c:+.3f}; corrupted full {ld_full_x:+.3f} split {ld_split_x:+.3f}; "
                 f"max |logit diff| over vocab {max_abs:.3f}; top-1 same {top_same}; tol {tol:.2f}")

    # 2: fork twice, golden untouched
    sig_before = cache_signature(cache_x)
    a, b = h.run_probe(corrupt, cache_x), h.run_probe(corrupt, cache_x)
    sig_after = cache_signature(cache_x)
    report.check(f"2 fork repeatable, golden untouched [{tag}]",
                 float((a - b).abs().max()) < EXACT_TOL and sig_before == sig_after,
                 f"max |diff| between two probe runs {float((a - b).abs().max()):.2e}; signature unchanged {sig_before == sig_after}")

    # 3: shapes
    text_cfg = h.inner.language_model.config
    shapes_ok, seen = True, {}
    for i, layer in enumerate(cache_c.layers):
        if isinstance(layer, LinearAttentionCacheLayerMixin):
            s = tuple(layer.recurrent_states[0].shape)
            seen.setdefault("M", s)
            shapes_ok &= s == (1, text_cfg.linear_num_value_heads, text_cfg.linear_key_head_dim, text_cfg.linear_value_head_dim)
            shapes_ok &= h.layer_types[i] == "linear_attention"
        else:
            s = tuple(layer.keys.shape)
            seen.setdefault("KV", s)
            shapes_ok &= s == (1, text_cfg.num_key_value_heads, clean.split, text_cfg.head_dim)
            shapes_ok &= h.layer_types[i] == "full_attention"
    report.check(f"3 cache shapes [{tag}]", shapes_ok, f"M {seen.get('M')} x{len(h.linear_layers)}, KV {seen.get('KV')} x{len(h.full_layers)}")

    # 4: self-patch no-op
    selfp = run_patched(h, clean, cache_c, [CachePatch("M", h.linear_layers, cache_c), CachePatch("conv", h.linear_layers, cache_c),
                                           CachePatch("KV", h.full_layers, cache_c)])
    report.check(f"4 self-patch no-op [{tag}]", float((selfp - split_c).abs().max()) < EXACT_TOL,
                 f"max |diff| {float((selfp - split_c).abs().max()):.2e}")

    # 5: all-swap ceiling
    allswap = run_patched(h, corrupt, cache_x, [CachePatch("M", h.linear_layers, cache_c), CachePatch("conv", h.linear_layers, cache_c),
                                               CachePatch("KV", h.full_layers, cache_c)])
    report.check(f"5 all-swap == clean [{tag}]", float((allswap - split_c).abs().max()) < EXACT_TOL,
                 f"max |diff| to clean probe {float((allswap - split_c).abs().max()):.2e}; LD {ld(h, allswap, pair):+.3f} vs clean {ld_split_c:+.3f}")

    # preview: the two arms
    arm_M = run_patched(h, corrupt, cache_x, [CachePatch("M", h.linear_layers, cache_c)])
    arm_KV = run_patched(h, corrupt, cache_x, [CachePatch("KV", h.full_layers, cache_c)])
    gap = ld_split_c - ld_split_x
    rec = lambda x: (ld(h, x, pair) - ld_split_x) / gap if abs(gap) > 1e-6 else float("nan")
    print(f"  preview [{tag}]: LD clean {ld_split_c:+.3f}, corrupted {ld_split_x:+.3f}; recovery all-M {rec(arm_M):.2f}, all-KV {rec(arm_KV):.2f}", flush=True)

    # 6: hook surfaces
    last = len(h.layers) - 1
    cap = Capture("residual", last)
    h.run_probe(clean, cache_c, hooks=[cap])
    over = h.run_probe(corrupt, cache_x, hooks=[Override("residual", last, cap.value)])
    res_ok = float((over - split_c).abs().max()) < EXACT_TOL
    fl = h.full_layers[len(h.full_layers) // 2]
    hcap = Capture("heads", fl)
    h.run_probe(clean, cache_c, hooks=[hcap])
    hover = h.run_probe(corrupt, cache_x, hooks=[Override("heads", fl, hcap.value, positions=[-1], heads=[0, 1])])
    head_changed = float((hover - split_x).abs().max()) > 0
    report.check(f"6 hook surfaces [{tag}]", res_ok and head_changed,
                 f"last-layer residual override max |diff| to clean {float((over - split_c).abs().max()):.2e}; "
                 f"heads 0-1 at layer {fl} override changed logits by {float((hover - split_x).abs().max()):.3f}")

    # 9: study-span embedding patch
    emb_c, emb_x = Capture("embeds", 0), Capture("embeds", 0)
    h.run_prefix(clean, hooks=[emb_c])
    h.run_prefix(corrupt, hooks=[emb_x])
    spans = study_spans(emb_c.value, emb_x.value, clean.split)
    repatched = h.run_prefix(corrupt, hooks=[Override("embeds", 0, emb_c.value, positions=spans["item"])])
    span_logits = h.run_probe(corrupt, repatched)
    span_diff = float((span_logits - split_c).abs().max())
    runs = contiguous_runs(spans["item"])
    report.check(f"9 study-span embedding patch == clean [{tag}]", runs == 1 and span_diff < EXACT_TOL,
                 f"item span {len(spans['item'])} tokens from {spans['item'][0]} in {runs} run(s), {len(spans['after'])} after; "
                 f"max |diff| to clean probe {span_diff:.2e}")
    del repatched

    # 7: backward through the kernel (single pass, small prompt)
    try:
        mid = len(h.layers) // 2
        grads = {}
        def hook(module, args, output):
            t = output[0] if isinstance(output, tuple) else output
            t.retain_grad()
            grads["t"] = t
        handle = h.residual_module(mid).register_forward_hook(hook)
        inputs = {k: v for k, v in clean.inputs.items() if k != "attention_mask"}
        with torch.enable_grad():
            logits = h.model(**inputs, position_ids=clean.position_ids, logits_to_keep=1).logits[0, -1].float()
            metric = torch.logsumexp(logits[h.ids_for(pair.answer)], 0) - torch.logsumexp(logits[h.ids_for(pair.counterfactual)], 0)
            metric.backward()
        handle.remove()
        g = grads["t"].grad
        report.check(f"7 backward through kernel [{tag}]", g is not None and torch.isfinite(g).all(),
                     f"grad at layer {mid} residual: shape {tuple(g.shape)}, |g| max {float(g.abs().max()):.3e}")
        h.model.zero_grad(set_to_none=True)
    except Exception as e:
        report.check(f"7 backward through kernel [{tag}]", False, f"{e.__class__.__name__}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--trials", type=int, default=2)
    ap.add_argument("--datasets", nargs="+", default=["Brady2008", "tokens"])
    ap.add_argument("--out", default="results/interp/phase0_checks.json")
    args = ap.parse_args()

    t0 = time.time()
    ev = build_evaluator({"name": args.model.split("/")[-1].lower(), "model_id": args.model, "device_map": "cuda"})
    h = Harness(ev)
    print(f"{args.model}: {len(h.layers)} layers, linear {h.linear_layers}, full {h.full_layers}; load {time.time() - t0:.0f}s", flush=True)

    report = Report()
    for dataset in args.datasets:
        for trial in range(args.trials):
            pair = pairs.make_pair(ev, "afc_recognition", dataset, args.n, seed=0, trial=trial)
            run_checks(h, pair, report, f"{dataset} n={args.n} t={trial}")
            torch.cuda.empty_cache()

    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"\n{report.failures} failures of {len(report.rows)} checks; peak GPU {peak:.1f} GB; {time.time() - t0:.0f}s", flush=True)
    from pathlib import Path
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"args": vars(args), "checks": report.rows, "peak_gpu_gb": peak}, indent=2))
    raise SystemExit(report.failures)


if __name__ == "__main__":
    main()
