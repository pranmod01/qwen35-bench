"""Phase 0 check 8: can nnsight wrap the Qwen3.5 hybrid under transformers 5.17 and read/set a residual?

nnsight is installed outside the benchmark env (pip --target) and put on PYTHONPATH by the caller, so a failure
here never touches the env. Usage (in scripts/interp.sbatch): PYTHONPATH=<side dir> python -m interp.phase0.test_nnsight
"""
import argparse
import time

import torch

from bench.models import build_evaluator
from interp import pairs
from interp.harness import Capture, Harness


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--layer", type=int, default=10)
    args = ap.parse_args()

    try:
        import nnsight
        print(f"nnsight {nnsight.__version__}", flush=True)
    except Exception as e:
        print(f"[FAIL] import nnsight: {e.__class__.__name__}: {e}")
        raise SystemExit(1)

    ev = build_evaluator({"name": "test", "model_id": args.model, "device_map": "cuda"})
    h = Harness(ev)
    pair = pairs.make_pair(ev, "afc_recognition", "tokens", 5, seed=0, trial=0)
    prompt = h.prepare(pair.clean)
    inputs = {k: v for k, v in prompt.inputs.items() if k != "attention_mask"}

    # Reference from a plain hook.
    cap = Capture("residual", args.layer)
    with h.hooked([cap]):
        ref_logits = h.full_pass(prompt)
    ref = cap.value

    t0 = time.time()
    wrapped = nnsight.NNsight(ev.model)
    layer = wrapped.model.language_model.layers[args.layer]
    with torch.no_grad():
        with wrapped.trace(**inputs, position_ids=prompt.position_ids, logits_to_keep=1):
            got = layer.output.save()
            logits = wrapped.output.logits.save()
    got = got[0] if isinstance(got, tuple) else got
    read_ok = torch.allclose(got.float(), ref.float(), atol=1e-2)
    print(f"[{'PASS' if read_ok else 'FAIL'}] nnsight read of layer {args.layer} residual matches the hook "
          f"(max |diff| {float((got.float() - ref.float()).abs().max()):.2e}); trace {time.time() - t0:.1f}s", flush=True)

    # Set: overwrite the residual with zeros at the last position and confirm the logits change.
    with torch.no_grad():
        with wrapped.trace(**inputs, position_ids=prompt.position_ids, logits_to_keep=1):
            out = layer.output
            target = out[0] if isinstance(out, tuple) else out
            target[:, -1] = 0
            logits2 = wrapped.output.logits.save()
    changed = float((logits2[0, -1].float() - ref_logits).abs().max())
    print(f"[{'PASS' if changed > 0 else 'FAIL'}] nnsight set of layer {args.layer} residual changed the logits by {changed:.3f}", flush=True)
    raise SystemExit(int(not (read_ok and changed > 0)))


if __name__ == "__main__":
    main()
