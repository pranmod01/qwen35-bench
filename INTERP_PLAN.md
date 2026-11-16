# Mechanistic interpretability phase: plan

Drafted 2026-11-09. The benchmark phase showed that every Qwen3.5 model loses recognition, serial order and
paired-associate memory as the study list grows. This phase asks *where* that memory lives inside the model and
*where it breaks*, by activation patching, and then uses the answer to try fixes.

## 1. The one question, asked at every grain

Every experiment here has the same shape:

1. Build two versions of a trial that differ in exactly one thing (section 4). In the **clean** version the
   correct answer is X. In the **corrupted** version the correct answer is the opposite, Y.
2. Run both. Take one piece of the clean run's memory and copy it into the corrupted run, leaving everything
   else corrupted. Ask the probe question again.
3. If the answer moves from Y back toward X, that piece of memory carried what the task needs.

The *piece* is what varies. The coarsest version is the headline experiment: copy **all** DeltaNet states
(the "notebook"), or copy **all** softmax keys and values (the "library"), and see which one brings the answer
back for each task. After that the same swap is done per layer, per attention head, and per study position, to
localise within whichever surface carried the task. The reverse direction (corrupted piece into the clean run,
"is it necessary" rather than "is it sufficient") is run alongside every time.

Swapping is preferred over zeroing throughout. Where an ablation is ever needed, use the mean or a resampled
activation, never zeros.

## 2. The models

Qwen3.5 is a hybrid: three Gated DeltaNet blocks, then one Gated Attention (ordinary softmax) block, repeated.
Full-attention layers are at index 3, 7, 11, ... (0-based). Every block is followed by an MLP.

| model | layers | DeltaNet (L) | full attention (F) | hidden | DeltaNet state per layer | attention heads (query / key-value), head dim |
|---|---|---|---|---|---|---|
| 0.8B | 24 | 18 | 6 | 1024 | 16 heads x 128 x 128 | 8 / 2, 256 |
| 2B | 24 | 18 | 6 | 2048 | 16 x 128 x 128 | 8 / 2, 256 |
| 4B | 32 | 24 | 8 | 2560 | 32 x 128 x 128 | 16 / 4, 256 |
| 9B | 32 | 24 | 8 | 4096 | 32 x 128 x 128 | 16 / 4, 256 |
| 27B | 64 | 48 | 16 | 5120 | 48 x 128 x 128 | 24 / 4, 256 |

**What a DeltaNet layer remembers.** Each layer carries one matrix per value head, `M`, of size 128 x 128.
Reading each token it updates `M_t = a_t * M_{t-1} + b_t * (v_t - M_{t-1} k_t) k_t^T`: decay the old matrix a
little (`a_t`), look up what it already holds for this key, and write only the difference (`b_t`). Reading is
`M_t q_t`: multiply by the probe's query and get back a mix of stored values weighted by key similarity. `M` is
the same size whether 5 or 27,000 items were studied; it holds the sum of all writes, not a record per item.
This is the fixed-size "notebook", and the recall-capacity theory (Zoology, Based) says it is where long-list
recall should fail.

**What a full-attention layer remembers.** The key and value vectors of every token seen so far, kept
separately. Grows with the list. Exact lookup by content is possible. This is the "library".

**Where these live in the HuggingFace code** (`transformers 5.17.0`, `models/qwen3_5/modeling_qwen3_5.py`):

- `Qwen3_5DecoderLayer` holds `.linear_attn` (DeltaNet) or `.self_attn` (attention), then `.mlp`. The decoder
  layer's output is the residual stream. The mixer's output (after `out_proj` / `o_proj`) is the block output.
  The input to `o_proj` is the per-head attention output, after the sigmoid output gate.
- The DeltaNet state is computed by the `flash_linear_attention` Triton kernel (`chunk_gated_delta_rule`,
  version 0.5.2 installed). The kernel does not store `M` at every position. It returns `M` only at the end of
  the pass, and only when a cache object is passed. With a cache, `M` sits in
  `cache.layers[i].recurrent_states[0]`, shape `[1, heads, 128, 128]`, fp32. The layer's short causal
  convolution keeps its last three inputs in `conv_states[0]`.
- Full-attention layers keep `.keys` and `.values` in the same cache object, shape `[1, kv_heads, tokens, 256]`.
- A forward call given a prefilled cache continues the computation exactly (the chunked kernel takes
  `initial_state`). The cache is updated in place, so a probe run changes the cache it was given.

## 3. Method: patch the cache between prefix and probe

Every trial's prompt splits into two parts:

- **Prefix**: the study list, the model's "I have studied the sequence" turn, and the opening of the test turn.
  Clean and corrupted differ only inside the prefix.
- **Probe**: the test question, the options, the generation prompt and the `Answer: ` prefill. Identical
  token-for-token in both versions.

Run the prefix once with `use_cache=True` and keep the returned cache. The cache is the only channel through
which the study list can influence the probe, so by construction it *is* the model's memory of the list: the
`M` matrices, the keys and values, and the three-token conv states. Then run the probe from an edited copy of
the cache. Per trial:

1. Clean prefix forward, keep cache A. Corrupted prefix forward, keep cache B.
2. Fork B: clone its `M` matrices (small: 4B is 24 layers x 2 MB) and conv states; share its key/value tensors
   by reference (the probe's `torch.cat` makes new tensors, so B stays intact). Overwrite the chosen entries
   with A's.
3. Run the probe from the fork with `logits_to_keep=1`. Read the metric.

**Why this design.**

- The prefix is the expensive forward and runs once per trial per version. Each patch is a probe forward of a
  few hundred tokens, under a second. A sweep over all layers, heads and both directions is seconds, even with
  an 80,000-token study list. Attribution patching, whose purpose is to avoid one long forward per location,
  is therefore not needed for the cache surfaces.
- "`M` as read at the probe" is the natural first object: the probe's own writes happen after the patch, so
  there is no propagation problem to correct for.
- Keys/values and `M` are the same kind of object in the same cache, so the notebook-versus-library comparison
  is a one-line difference in which entries are swapped.

**What it does not tell you, and the slower experiments that do.**

- Boundary patching says what the probe *reads from*, not where along the study list the information was
  written or lost. Two additions cover that:
  - **State trajectory**: run the prefix in chunks and snapshot `M` after each chunk. One prefix forward in total
    gives `M` across the whole list, which is how "where does the notebook start smearing" is measured.
  - **Residual stream at a study position**: overwrite the residual at (layer, position span of the manipulated
    item) and re-run the prefix. One prefix forward per (layer, span). For the 4B in its decay zone that is
    0.2–3 s each, so a 32-layer sweep over a few spans is minutes. Attribution patching is kept as an optional
    triage for this surface only.
- Keys and values at full-attention layer 7 were computed from a residual stream that already contains layers
  0–6, three of which are DeltaNet. "Clean keys/values" are not pure softmax memory. The per-layer and per-head
  swaps are what sharpen that reading, and every localisation claim is checked against a second surface.

**Surfaces, from coarse to fine.**

| surface | what is swapped | how |
|---|---|---|
| all `M` | every DeltaNet layer's state | cache edit |
| all K/V | every full-attention layer's keys and values | cache edit |
| `M` by layer | one DeltaNet layer's state | cache edit |
| K/V by layer | one full-attention layer's keys and values | cache edit |
| attention heads | one head's output at one layer, probe positions | pre-hook on `o_proj` input |
| residual at probe | decoder layer output at every probe position (question, options, answer prefill) | forward hook, probe run |
| residual at study span | decoder layer output over the manipulated item | forward hook, prefix re-run |

Patching M "by head" is also possible (one 128 x 128 slice), and is the natural follow-up when a layer lights up.

**Metric.** Logit difference at the answer position:
`logsumexp(logits[ids of X]) − logsumexp(logits[ids of Y])`. For recognition and serial order the ids are the
digit tokens for "1" and "2" after the `Answer: ` prefill (the benchmark's `Qwen35Evaluator._choice_token_ids`
already handles both tokenisations). For paired associates they are the first tokens of the two words after the
generation prompt (the 4B answers with the bare word on 46–48 of 50 trials at n ≤ 25, so the bare form's first
token is the answer token; the spaced form's is included too, as the digit scoring does). The counterfactual
word is drawn from the pool with the same token count as the target and no shared first token. Every patch is reported as **recovery** = `(LD_patched − LD_corrupted) / (LD_clean − LD_corrupted)`,
averaged over trials, with the clean baseline, the corrupted baseline and the all-swap ceiling on every plot.

## 4. Clean/corrupted pairs

Exactly one thing differs, positions stay aligned, and the correct answer flips. Aligned positions come for free:
a token item is replaced by another item of the same length, a Brady or ImageNet image by another image of the
same size (both are 256 px, 66 tokens each). THINGS images vary in size and are not used for patching.

| task | clean prefix | corrupted prefix | probe (shared) | answer |
|---|---|---|---|---|
| recognition | target T studied at slot s; foil F never shown | F studied at slot s; T never shown | "which of these two was in the sequence" with T and F as the options | 1 ↔ 2 |
| serial order | A at slot i, B at slot j > i | B at slot i, A at slot j | "which appeared first" with A and B as the options | 1 ↔ 2 |
| paired associates | pair (item, WORD_B) at slot s | pair (item, WORD_R) at slot s, R with the same token count | "which word was shown with this item" | B ↔ R |

Slot `s` and lag `j − i` are design variables; the benchmark's trial builders (`tokens/stimuli.py`,
`tasks/`) already record serial position and lag. A second recognition control replaces T with an unrelated
item instead of with F; it is kept for Phase 3 but is not the default because its corrupted answer is undefined.

## 5. Where it runs: model, modality, sizes

**Primary model: the 4B. Replication: the 9B.** The 0.8B and 2B answer "1" on most trials at small list sizes,
so they have no clean ceiling to anchor on. The 27B is at ceiling nearly everywhere we can run it and needs two
cards.

**Modality: Brady images first, token items as the control.** Images are what the project is about; Brady
images are uniform (66 tokens) so positions align. Token items (`tokens` dataset, item length 8) give the same
tasks with no vision encoder and are the check that a finding is about memory rather than vision.

Easy slice and decay zone, mean accuracy over seeds (chance 0.5 for recognition and serial order, ~1/n for PAM):

Brady images (`results/recognition_single_foil`, `serial_order_final`, `pam_final`):

| task | model | at ceiling through | decay zone | notes |
|---|---|---|---|---|
| recognition | 4B | n=250 (0.98) | 500 (0.93) → 2399 (0.60) | 2399 images ≈ 160k tokens |
| recognition | 9B | n=500 (0.98) | 1000 (0.97) → 2399 (0.86) | |
| serial order | 4B | n=5 (0.97) | 25 (0.88) → 100 (0.60) | chance from 250 |
| serial order | 9B | n=25 (0.94) | 100 (0.62) → 250 (0.64) | chance from 500 |
| paired associates | 4B | n=500 (0.94) | 1000 (0.76) → 1637 (0.68) | |
| paired associates | 9B | n=500 (0.90) | 1000 (0.76) → 1637 (0.71) | |

Token items, length 8 (`results/tokens_full`, `tokens_pam`):

| task | model | at ceiling through | decay zone | chance from |
|---|---|---|---|---|
| recognition | 4B | n=25 (0.93) | 100 (0.85) → 1000 (0.57) | 10,000 |
| recognition | 9B | n=2,500 (1.00) | 10,000 (0.90) → 27,000 (0.88) | not reached |
| serial order | 4B | n=5 (0.97) | 25 (0.73) → 250 (0.68) | ~1000 |
| paired associates | 4B | n=100 (0.91) | 250 (0.79) → 1500 (0.40) | 4000 |
| paired associates | 9B | n=25 (0.95) | 100 (0.81) → 500 (0.46) | 4000 |

The 4B's decay zones sit inside 1,000–100,000 tokens of context, which one 48 GB card handles (the 4B used
39 GB at 144k tokens in the perception checks).

## 6. Code layout

A new top-level package, parallel to `tokens/`, laid out by phase. Code shared by every phase sits at the top:

```
interp/
  harness.py          shared: load the model via the benchmark's Qwen35Evaluator (reuses _prepare_inputs and
                      _choice_token_ids); run a prefix, hold the golden cache, fork it; run the probe; cache
                      patches (M, conv, K/V) and hooks (heads, residual); logit_diff
  pairs.py            shared: clean/corrupted builders for the three tasks x {Brady, ImageNet, tokens}, on top of
                      tokens/stimuli.py and the tasks' build_messages
  phase0/
    test_harness.py   substrate checks 1-7, one sbatch job
    test_nnsight.py   substrate check 8 (nnsight)
  phase2_3/           Phases 2 and 3 run the same sweep; only the config differs
    run_patch.py      config-driven sweep, one JSON per cell with the config and provenance embedded as the
                      benchmark does; existing cells are skipped so a job resumes
    plot_patch.py     recovery by surface, by layer, by head, by list size
```

| phase | code | configs | outputs |
|---|---|---|---|
| 0 substrate | `interp/phase0/` | none (command-line arguments) | `results/interp/phase0_checks.json` |
| 1 characterise | `bench/plot_decay.py` (benchmark data only) | none | `plots/decay/decay_curves.png` |
| 2 easy slice | `interp/phase2_3/` | `configs/interp/phase2/` | `results/interp/phase2_*`, `plots/interp/phase2_*` |
| 3 decay zone | `interp/phase2_3/` | `configs/interp/phase3/` | `results/interp/phase3_*`, `plots/interp/phase3_*` |

Config, one file per experiment:

```yaml
run_name: patch_recognition_brady_4b
seeds: [0]                                    # one seed; the benchmark used two
model: {name: qwen3.5-4b, model_id: Qwen/Qwen3.5-4B}
task: afc_recognition
stimuli: {dataset: Brady2008}                 # or {dataset: tokens, token_pool: clean, item_length: 8}
n_images: [1, 5, 25, 100, 250, 500, 1000, 1500, 2000, 2399]   # the benchmark's Brady grid
n_trials: 50                                                  # as every benchmark cell
surfaces: [all, all_M, all_conv, all_KV, M_by_layer, KV_by_layer, heads_by_layer, residual_at_probe]
directions: [clean_to_corrupt, corrupt_to_clean]
```

(`all` is the ceiling: M, conv state and K/V together. `all_conv` is the three-token convolution state on its
own; it is not identical across clean and corrupted because those tokens' projections depend on the earlier
context, so it is swapped explicitly rather than assumed shared.) The first two configs are
`configs/interp/phase2/brady_4b.yaml` and `tokens_4b.yaml`; `python -m interp.phase2_3.run_patch <config> --dry-run` lists
the cells.

`residual_at_study_span` and `state_trajectory` get their own configs because they re-run or chunk the prefix.
Results go under `results/interp/<run_name>/`. The benchmark code is not changed; Phase 4 may add prompt
layouts to the task modules as config options.

Harness library: plain PyTorch hooks plus cache manipulation is the baseline, because the two most important
surfaces live in the cache object and not in any module output. nnsight is tried in Phase 0 (check 8) for the
module surfaces (residual stream, head outputs); if it imports and traces the 4B under transformers 5.17 it is
used there, otherwise hooks do that job too.

## 7. Phases and gates

**Phase 0: substrate (days).** `interp/phase0/test_harness.py` as one sbatch job on the 4B, n=25 Brady and n=25 tokens:

1. Prefix + probe from the cache reproduces the single-pass logits within bf16 tolerance, for an image prompt
   (the probe carries its own two images; multimodal position ids must continue from the prefix) and for a
   token prompt.
2. Forking a cache and running the probe twice gives identical logits and leaves the golden cache unchanged.
3. Shapes: `M` is `[1, 32, 128, 128]` at the 24 DeltaNet layers; keys/values `[1, 4, T, 256]` at the 8 others.
4. Self-patch is a no-op: the clean cache with its own `M` and K/V swapped in gives the clean logits exactly.
5. All-swap ceiling: the corrupted cache with every `M` and every K/V from clean gives the clean answer.
6. Head hook on the `o_proj` input and residual hook on a decoder layer output both read and overwrite.
7. A backward pass through the DeltaNet kernel runs at n=25 (needed only for attribution patching).
8. nnsight: install into a copy of the env, trace one 4B forward, read and set a residual. Pass or fall back.

Gate: checks 1–6 pass on both modalities.

**Result (2026-11-10, job 13854750, `results/interp/phase0_checks.json`):** passed. Checks 2–7 exact on all
four trials (two Brady, two tokens, n=25); check 1 agrees to within bf16 noise (logit differences within 0.25
of the single pass, same top-1; the clean Brady run matched exactly). Peak GPU 31.7 GB on the 4B, most of it
check 7's backward over an 1,863-token prompt. The preview of the two-arm swap, clean pieces into the corrupted
run, recognition at n=25:

| trial | LD clean | LD corrupted | recovery all-M | recovery all-K/V |
|---|---|---|---|---|
| Brady t0 | +3.75 | −6.38 | 0.01 | 0.96 |
| Brady t1 | +3.88 | −5.13 | 0.03 | 0.99 |
| tokens t0 | +2.63 | −0.38 | 0.00 | 0.96 |
| tokens t1 | +1.38 | −3.00 | −0.23 | 0.91 |

Four trials only, but the same answer on both modalities: at this size recognition is read from the softmax
library, not the notebooks. Phase 2 tests whether that holds over 50 trials and across the three tasks.

**Phase 1: characterise (done 2026-11-10).** No new runs: `python -m bench.plot_decay` draws the benchmark's
accuracy against list size per (model, task, modality) in one figure, `plots/decay/decay_curves.png`, with a
logistic line as a guide for the eye. The list sizes where each model is at ceiling and where it decays are the
section 5 tables; those fix the Phase 2 and Phase 3 grids.

**Phase 2: validate on the easy slice (1–2 weeks).** 4B, Brady, n=5 and 25 for recognition and PAM, n=2 and 5
for serial order, where the model is at ceiling and the memory is certainly present. Full surface sweep in both
directions, 50 trials per cell and one seed, matching the benchmark's cells. Then the same on tokens.

- Gate A: the all-swap ceiling recovers the clean answer on every task, and at least one of all-`M` /
  all-K/V recovers most of it. If neither partial swap recovers anything, the harness is suspect, not the model.
- Gate B (only if attribution patching will be used): its ranking of residual spans matches real patching at
  n=25. If not, drop it; real patching is affordable here.
- A synthetic associative-recall task is only built if the 4B cannot be put at ceiling, which section 5 says
  it can.

**Result, Brady (2026-11-11, job 13855431, `results/interp/phase2_brady_4b/`, figures `plots/interp/phase2_brady_4b/`):**
Gate A passes on every cell, in both directions, 50 trials each. The ceiling swap recovers 1.00 everywhere.

| task, n | LD clean / corrupted | all-M | all-conv | all-K/V | top K/V layer | top head |
|---|---|---|---|---|---|---|
| recognition, 5 | +3.6 / −3.9 | 0.01 | 0.00 | 0.97 | L11 0.31 | L15H1 0.22 |
| recognition, 25 | +4.4 / −4.3 | 0.00 | 0.00 | 1.00 | L11 0.38 | L31H14 0.16 |
| serial order, 2 | +4.0 / −3.3 | −0.01 | −0.01 | 0.98 | L11 0.53 | L19H11 0.23 |
| serial order, 5 | +1.9 / −2.3 | 0.02 | 0.01 | 1.03 | L11 0.61 | L11H8 0.38 |
| paired associates, 5 | +13.0 / −13.1 | 0.03 | 0.00 | 0.97 | L27 0.48 | L31H14 0.21 |
| paired associates, 25 | +11.2 / −11.5 | 0.02 | 0.00 | 0.99 | L27 0.34 | L31H14 0.23 |

(clean → corrupted direction; the reverse direction gives the same numbers to within 0.05.)

Reading: on the easy slice every task is read from the softmax K/V, none from the DeltaNet states or the
convolution states. Within the K/V, no single layer carries the answer: recognition and serial order lean on
layer 11 (with 7 and 15), paired associates on layer 27 (with 19, 23, 31), so the two kinds of task use
different full-attention layers. The residual-at-probe sweep shows where the answer is written into the
probe's stream: by layer 15–18 for recognition and serial order, only by layer 31 for paired associates
(0.87–0.96 from layer 28). Single heads recover at most 0.2–0.4, so retrieval is spread over several heads per
layer. The 20–25 min per cell came almost entirely from the 128 head-level probe runs per direction.

**Result, tokens (2026-11-11, jobs 13859405 and 13860280, `results/interp/phase2_tokens_4b/`):** the same
picture on the control modality, so it is about memory, not vision.

| task, n | LD clean / corrupted | all-M | all-conv | all-K/V | top K/V layer | top head |
|---|---|---|---|---|---|---|
| afc_recognition, 25 | +2.0 / -2.1 | 0.00 | -0.01 | 1.03 | L7 0.29 | L15H1 0.29 |
| afc_recognition, 5 | +1.6 / -1.7 | -0.08 | 0.01 | 1.29 | L7 0.43 | L19H2 0.23 |
| paired_associates, 25 | +12.9 / -12.7 | 0.01 | 0.00 | 0.99 | L27 0.38 | L31H14 0.22 |
| paired_associates, 5 | +13.9 / -12.8 | 0.02 | 0.00 | 0.97 | L27 0.43 | L31H14 0.22 |
| serial_order, 2 | +1.0 / -2.4 | 0.04 | -0.00 | 0.96 | L11 0.59 | L19H11 0.37 |
| serial_order, 5 | +1.1 / -1.5 | 0.07 | -0.00 | 0.97 | L11 0.52 | L15H11 0.25 |

Token recognition leans on layer 7 rather than 11 and its logit gaps are small (about 2 either side), so its
recovery ratios are noisier (1.29 at n=5 is the clean K/V overshooting a small gap). Serial order and PAM pick
the same layers and the same top heads (L19H11, L31H14) as on images. The first token-PAM job failed its
alignment check on some trials: in the sandwich layout the word's second copy runs into the next "Word:"
label and the tokenizer can merge across that boundary differently for the two words. The builder now renders
both prompts and keeps the first candidate word that aligns (0 misalignments in 150 pairs), and the runner
retries cells whose file holds an error.

**Result, 9B replication (2026-11-14, jobs 13909054 and 13909055, `results/interp/phase2_{brady,tokens}_9b/`):**
the same picture. The 9B has the 4B's depth (32 layers, attention at 3, 7, …, 31) and head counts (16 query / 4
key-value, 32 DeltaNet heads); it is wider (hidden 4096 vs 2560, MLP 12288 vs 9216), so layers and heads compare
one to one. All-K/V recovers 0.93–1.01 and all-M 0.00–0.06 in every cell (0.12 on token recognition n=5, gap ±1.3).
Peak K/V layers match the 4B: L11 for recognition and serial order (L7 with L3 and L11 for token recognition), L27
for paired associates; the answer is settled in the residual by the same layers (L11–15, L27). Single-layer peaks are
mostly a little lower on the 9B (Brady recognition n=25 L11 0.26 vs 0.38, Brady serial order n=5 0.47 vs 0.61), so it
spreads the readout over more layers; token serial order is the exception (L11 0.71 vs 0.52).

**Phase 2 verdict:** gate A passed on both modalities. Open: whether the same picture holds into the decay
zone (Phase 3), where the notebooks may start to matter or the K/V readout may fail.

**Phase 3: the sweep (bulk).** Same configs, n extended over the benchmark's own grids (Brady recognition 1 … 2399, serial order
2 … 2400, PAM 1 … 1637; tokens 1 … 27,000 where one card allows), so patching cells line up with accuracy cells. Order: recognition, then serial order, then paired
associates. For each task and n: which surface recovers, which layers, and how recovery changes as accuracy decays. The
per-head sweep (`heads_by_layer`) is left out of the Phase 3 configs for now (decided 2026-11-14): in Phase 2 it
only refined the layer picture and was most of the run time (128 extra probe runs per direction per trial). It
is added back for the cells where the layer results are worth localising further. Add the state trajectory and the residual-at-study-span map at two or three n. Later, for those
cells, the retrieval-head detector on the full-attention layers: attention mass from the answer position onto the target's
slot versus the rest, per head (eager attention to read the weights), confirmed by swapping those heads'
outputs. Validate every localisation with a second surface. Aim for a dissociation across tasks (for example,
K/V restores recognition but not serial order, `M` the reverse). 9B replication on the cells where the 4B shows
an effect. Headline figure: recovery by surface x task x n.

**Phase 4: localise → fix.** Chosen by what Phase 3 finds. Candidate fixes, all reachable from this harness or
from a config change in the benchmark:

- **Restrict full attention to a sliding window** at probe time (a 4D mask on the `full_attention` entry of the
  mask dict). Tests whether recognition needs global softmax retrieval. Swap, not ablate.
- **Vary which full-attention layers are active** for the probe, or how many.
- **Just Read Twice (JRT):** a causal model must decide what to store before it sees the probe. Two prompt
  layouts test whether that is the failure: `repeat` (the study list shown twice) and `query_first` (the
  options, or the PAM cue, shown before the list). Ordinary benchmark runs, added as `prompts:` options to the
  three task modules. If either extends the list size at which accuracy holds, the failure is storage selection
  and the fix needs no training.
- **Widen or retrain the recurrent state**: only through fine-tuning; the benchmark's `layer_variant` hook
  already warns that converted layers are randomly initialised.
- Baseline to differentiate from: Stable-Mamba (activation-subspace bottlenecks fixed architecturally).

## 8. Compute and workflow

- One 48 GB card runs the 4B through its full decay zone. Phase 2 is hours of card time, Phase 3 days. The 9B
  needs two or three cards above ~10k tokens (the `tokens_full` layouts).
- Everything runs as sbatch jobs with a config; the login node kills long shell jobs. Results under
  `results/interp/`, small JSONs. Home quota is 9.6 GB today; check before Phase 3.
- Standing exclude list for nodes that kill jobs at launch: ins037, ins047, ins058, ins082, ins091, ins092.

## 9. Pitfalls to design against

- **Non-uniqueness**: in recurrent models the causal set is not unique; a near-disjoint set of heads can match.
  Never claim "the memory lives here" from one surface.
- **Subspace illusion**: a partial patch can work by waking a dormant pathway. Always confirm with a second
  intervention surface.
- **Attribution scores do not track real patching** for large or saturated effects. Triage only.
- **Propagation**: a patch at a study position changes every later state. Report those as whole-prefix
  reruns, never as isolated sites.
