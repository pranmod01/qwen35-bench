# Limitations and known constraints

Findings from setting up and running the benchmark (2026-09-17 to 2026-09-28). Unless noted, measurements used an
RTX A6000 (48 GB), torch 2.10.0+cu128, transformers 5.17.0, bf16, greedy decoding, and images capped at 512 px.
The diagnostic scripts are not in the repo; they are in
`/insomnia001/depts/zgroup/zgroup_burg/zgroup/users/pm3361/tmp/` (`perception_test.py`, `image_diag.py`,
`max_images.py`, `diag_choice.py`, `control_2afc.py`), with logs under `logs/`.

The capacity sections below are perception checks (images and question in the same prompt), not memory
tasks. The memory tasks are harder, so expect them to degrade well before these limits.

## Qwen3.5-0.8B's image comparison is unreliable and stimulus-dependent

The perception test labels each image ("Image 1:", "Image 2:", ...) and asks whether a probe image is the same
as Image k, where k is random; the probe matches on half the trials, so chance is 50%. With 20 trials per size:

| Images | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3.5-0.8B | 60% | 55% | 50% | 80% | 55% | 60% | 45% | 30% | 45% |
| Qwen3.5-4B | 100% | 100% | 100% | 100% | 100% | 90% | 95% | 100% | 100% |

(jobs 12729481, 12729482)

- With 20 trials, differences of about ±15 points are noise.
- The 0.8B still perceives images deep in a prompt: asked what the first of 256 images showed, it answered
  correctly (job 12729420). It can name objects; it compares them only in the easy regime (below).

### Which comparisons the 0.8B can do (2026-09-24, 100 episodes per cell)

The perception test above used THINGS images shrunk to 256 px, and suggested the 0.8B simply cannot compare
images. The full 2-AFC runs are more specific: with a single studied image it scores **0.90 on THINGS** and
**0.71 on Brady2008** — and the Brady number is an artifact, because it answered "2" on 83% of trials
(1.00 when "2" was correct, 0.37 when "1" was). Its real discrimination is near-perfect on THINGS and
at chance on Brady.

Resolution does not explain the split (job 12867560, `configs/archive/img_resolution.yaml`): the same model on
THINGS at n=1 scored 0.94 at 256 px versus 0.92 at 512 px, with balanced answers at both. Brady is natively
256 px and scored identically under both settings, confirming the manipulation only affected THINGS. At
256 px a THINGS image costs about the same ~68 tokens as a Brady image, so neither pixel count nor visual
token count is the cause. What is left is the images themselves: Brady objects are isolated on a white
background and share a layout, while THINGS photos differ in colour, background and composition at once.

- Treat 0.8B scores as a floor **and always read its answer distribution** — where it cannot discriminate it
  falls back to one answer, which scores near the target split (~0.5–0.7) and reads like partial memory.
- A forced-choice control (20 trials per cell, `control_2afc.py`) verified the scoring itself: when the study
  sequence contained neither test image, both the 2B and 4B dropped to 0.45–0.65 and answered "1" on 18–20 of
  20 trials, while real trials stayed at 0.85–1.00 with balanced answers. Repeated at n=500 (job 12913485,
  20 trials per cell): the 4B scored **1.00 / 1.00** (Brady / THINGS) on real trials against 0.60 / 0.40 on
  the control, so its recognition ceiling is real; the 2B is at chance either way (0.50 / 0.45 real).

### The model's own statements about what it sees are unreliable

Given one image repeated N times and asked to count them, the 0.8B answered "I can't see any images" from
N = 32 up (job 12664001). Checks showed the images were present and correctly encoded: prompt tokens scale
at 66 per Brady image, and vision features for an image are the same alone as in a batch of 128 (differences
of 0.016, which is bf16 rounding). The refusal came from the prompt, not a processing limit. Don't use
"can you see the images" style prompts to measure capacity.

## How many images fit in one prompt

Qwen3.5-4B, perception test, 5 trials per size (job 12729769):

| Images | Prompt tokens (approx.) | Accuracy | Peak GPU memory | Time per trial |
|---|---|---|---|---|
| 512 | ~36k | 5/5 | 16.6 GB | 19 s |
| 1,024 | ~72k | 5/5 | 24.0 GB | 20 s |
| 2,048 | ~144k | 5/5 | 39.1 GB | 42 s |
| 3,500 | ~245k | out of memory on 48 GB | — | — |

The limits, in the order they are hit:

1. **GPU memory.** It grows by about 14–15 GB per 1,000 images for the 4B, so 3,500 images should need about
   60 GB. That fits on the 80 GB H100 or 96 GB RTX PRO 6000 nodes, but this is untested. A chunked vision
   encoder (as in vision-memory's old `qwen_evaluator.py`) may also cut the peak; also untested.
2. **Context length.** 262,144 tokens natively. Brady images are 256×256 px and cost about 66 tokens
   (about 70 with a text label), so about 3,700 fit. THINGS photos are larger and cost about 256 tokens
   at the 512 px cap, so about 1,000 fit.
3. **Extended context.** The model card says YaRN extends the context to about 1,010,000 tokens (about
   14,000 Brady images). This is supported in transformers via a config change, but untested here, and
   quality at that length is unknown.

The 0.8B went further on a 48 GB card: **3,601 images / ~248k tokens at 38 GB** (2026-09-26), close to the
native context limit.

Only 8 of the 4B's 32 layers are full attention and keep a cache that grows with the prompt. That cache is
about 32 KB per token, or 8.6 GB for a full 262k prompt. The rest of the peak is temporary memory for
processing the whole prompt at once.


### Per task, measured on an RTX A6000 (47.4 GB usable), 2026-09-26

Real episodes, both calls a trial makes, one trial per rung (`$ZROOT/tmp/capacity_probe.py`, jobs
12900577-79 and 12901174-76). Peak GB is the hardware-independent number; the ceiling for any other
card follows from it.

| Task | Images in prompt | 0.8B | 2B | 4B | Binding limit |
|---|---|---|---|---|---|
| MST | n + 1 | 576 (13.1 GB) | 576 (18.7) | 576 (24.2) | dataset (1,152 pairs, half are foils) |
| Color memory | n + 1, or n + 2 with the wheel | 540 (14.0) | 540 (19.9) | 540 (25.5) | dataset (540 images) |
| Assoc. inference, Brady image pairs | 4 x chains + 3 | 800 ch. (34.1) | 500 ch. (30.0) | 500 ch. (36.7) | GPU for 2B/4B, dataset for 0.8B |
| Assoc. inference, Brady word pairs | 3 x chains + 1 | 1,200 ch. (38.2) | 800 ch. (35.1) | 800 ch. (42.8) | GPU for 2B/4B, dataset for 0.8B |
| Assoc. inference, THINGS image pairs | 4 x chains + 3 | 250 ch. (42.1) | ~200 ch. | ~100 ch. | context at 250 ch. (258k of 262k tokens) |

Tokens per image at `max_image_size: 512`: Brady 66, MST 146, Brady2013ColorObjects 146, THINGS 258.
So MST and color memory are dataset-bound on every model and can be run edge to edge, while
associative inference is the only task where the GPU binds first.

The 2B and 4B THINGS rungs above 100 chains ran on a 102 GB RTX PRO 6000, not an A6000; their A6000
ceilings are derived from peak GB rather than measured. Provenance now records the GPU name for every
run, because the partition mixes 48 GB A6000/L40S with 80 GB H100 and 96 GB RTX PRO 6000 nodes and the
largest conditions sit near the 48 GB edge.

### Models above 4B (2026-09-27)

Measured on the three original tasks' prompt shape, n studied images + 2 test images
(`$ZROOT/tmp/capacity_probe_2afc.py`, jobs 12938524 and 12938735):

| Model | Weights | Brady n=500 (33k tok) | THINGS n=250 (65k tok) | THINGS n=500 (129k tok) |
|---|---|---|---|---|
| 9B | 18.82 GB | 26.2 GB | 33.3 GB | **OOM on 48 GB** (needs ~48 GB+) |
| 27B | 54.71 GB | 72.0 GB | 88.3 GB | **OOM on 2 x 48 GB** (needs ~110 GB) |

- The 9B fits a 48 GB A6000 everywhere except THINGS n=500. In the actual sweep that cell peaked at
  47.7-48.0 GB and only completed because the jobs happened to land on **H100 NVL (94 GB)** nodes; on
  an A6000 (47.4 GB usable) it fails, exactly as the probe predicted. The 9B results therefore come
  from different silicon than the 0.8B/2B/4B runs, which is recorded per run in provenance.
- **Multi-GPU sharding works.** `device_map` is now a model-config key (default `"cuda"`); `"auto"`
  lets accelerate shard across the visible GPUs. The 27B ran correctly across 2 x A6000 with choice
  mass 1.000 on every rung, so the same route is available for the 35B-A3B and 122B-A10B.
- Sharding is **uneven** -- accelerate fills the first card first -- so a total budget is not the
  usable limit: 88.3 GB of a 101.8 GB two-card budget still OOMed on the next rung because card 0 was
  full. Give a sharded model more cards than the total suggests. Concretely, 54.71 GB of 27B weights over two cards
  leaves ~27 GB on card 0, so Brady n=250/500 (72 GB total, fine on 2 x A6000) was rejected on 2 x L40S.
  Size multi-GPU jobs by per-card headroom.
- Model runs are not on uniform hardware: 0.8B/2B/4B on A6000, 9B on H100 NVL, 27B on L40S and H100
  pairs. Irrelevant to accuracy, but peak-GB and timing numbers are not comparable across models.
- Weights scale at about 2.1 GB per billion parameters (18.82 GB at 9B, 54.71 GB at 27B), which puts
  the 35B-A3B near 73 GB and the 122B-A10B near 255 GB. Both need `device_map: auto` and several cards.

### Largest study list per task, from the context (2026-09-29)

All five sizes share the 262,144-token context, the tokenizer and the image tokenisation (patch 16,
merge 2), so the ceiling is the same for every model. Measured without a GPU by building real prompts
with each task's own code, the free-text call's 100 generated tokens included, under both chat-template
versions (identical counts) -- `$ZROOT/tmp/context_probe.py`, job 12963688:

| Task | Brady2008 (66 tok/image) | THINGS (~257 tok/image) |
|---|---|---|
| Recognition, `novel` foils | **1,200** -- dataset (half the images are foils), 80k tokens | **927** -- dataset, 239k tokens |
| Serial order | **2,400** -- dataset, 159k tokens | **1,017** -- context (1,854 in the dataset) |
| Paired associates (70 tok/pair) | **1,637** -- word pool, 115k tokens | **1,002** -- context |

- Brady runs out of images (or words) long before the context; THINGS fills the context at ~1,000.
- Recognition here is the standard 2n-image episode. `single_foil` raises it to 2,399 Brady / 1,853 THINGS
  (see "Stimulus set sizes").
- **Nothing detects a prompt that exceeds the context.** Past 262k tokens the model silently runs on
  positions it was not trained on. Stay clear of the THINGS ceilings: the n>500 configs use 927 on
  THINGS for all three tasks, ~9% under the context for serial order and paired associates.
- GPU memory near full context (~258k tokens), measured by the capacity probe for the small models and
  extrapolated from per-run peaks for the large ones: 0.8B 42 GB, 2B 55 GB, 4B 62 GB, 9B ~78 GB,
  27B ~190 GB summed over cards. So at the THINGS ceiling the 2B-9B each need an 80-94 GB card, and
  the 27B probably needs more than one H100 node's two cards (uneven sharding makes it worse).
- Going past these needs either a longer context (YaRN, ~1M tokens per the model card, untested here)
  or smaller THINGS images (256 px costs ~66 tokens, like Brady, but changes the stimulus).

### Partitions and preemption

`short` has a 12 h limit, which the 27B exceeds: at THINGS n=250 one trial takes 111 s, so a single
100-episode cell is about 3 h. `burst` allows 14 days and includes the same GPU types. Both partitions
have `PreemptMode=REQUEUE`, so a long job can be interrupted; results are written per run, so a
resubmit resumes. A mass job failure across several nodes at once (six jobs, five nodes, exit code 7,
no traceback, 2026-09-24) is consistent with preemption and is not a code fault.

Do not pin a GPU model in `--gres` on this cluster: pinned requests sat at `ReqNodeNotAvail` for hours
while a generic `gpu:N` request was placed within seconds. For cells that need bigger cards, put them in
their own config rather than a `--dependency` mop-up job (one was cancelled `DependencyNeverSatisfied`
although its parents completed).

### RTX PRO 6000 (Blackwell) fails on very long prompts

On the RTX PRO 6000 nodes (`gres` `R6000`), ~238k-token prompts crash in the forward pass with
`cudaErrorIllegalAddress` (0.8B and 9B, THINGS 512 px n=927 preflight,
`configs/archive/xl_preflight_things.yaml`). The same prompts run on A6000 and H100, and the R6000 is fine
up to ~159k tokens, about what the largest final cells need (Brady at 2,400 images; THINGS-256 peaks
near 123k). Keep any longer-prompt job off those nodes. The 9B final runs used R6000.
## Scoring choice tasks: forced choice, not text parsing

The 2-AFC tasks (recognition, serial order) are scored from the model's probability of "1" vs "2" as the
first token after a prefilled `Answer: ` turn, not by parsing its reply. Parsing free text produced wrong
numbers in three ways, all visible in `results/archive_2026-09-21_pre_fix/`:

- A reply ending in "1" was scored unparseable because the reasoning contained "22 images" (so both digits
  appeared). Other replies scored *correct* for the same reason.
- Replies were truncated at `max_new_tokens=100` before the model reached its answer.
- At mid study sizes the 2B/4B abandon the "reply with only the digit" instruction and write a paragraph:
  up to 97% of trials in a cell. The old scores for those cells were mostly parser artefacts.

Two properties of the forced choice have to be checked rather than assumed, so every trial records them:

- **`choice_mass`** — how much probability "1" and "2" hold together. The choice is renormalised over the two,
  so a low mass means the answer came from the tail of a distribution that was mostly about something else.
  `top_tokens` (top 5) shows where it went. Measured cause: at n=5 the models often answer with a **serial
  position** instead ("5": 0.32 beating both "1" and "2"), which is why low mass clusters at that size.
- **Answer bias** — one answer dominating a cell. Expected wherever a model cannot discriminate.

`bench/summarize.py` flags cells automatically: `LOW_MASS` when more than 10% of trials fall below 0.9 mass,
`BIAS` when one answer exceeds 80% of trials. `LOW_MASS` questions the measurement of those trials (the 4B's
serial-order n=5 cells are flagged and still score 0.88–0.96); `BIAS` questions whether the accuracy means
anything. Accuracy itself is always raw — no trials are dropped or reweighted — and `--simple` prints
accuracy alone.

The free-text reply is still generated and stored (`free_response`, `free_parsed`) so format compliance stays
reportable, but it no longer affects any score.


## Answer layouts for tasks whose answers are words

MST and named color memory ask for a word, not a digit, and the forced choice does not transfer
directly. Asked for `old`/`similar`/`new` or a color name, the models hold only **0.07-0.86** of their
next-token mass on the answer set, against ~1.0 for the digit tasks. Case-folding the token ids
recovers almost nothing (0.443 -> 0.487 on the 0.8B); the mass genuinely goes elsewhere
(`$ZROOT/tmp/diag_word_choices.py`, jobs 12901178-79):

- **digits** - the 4B answers MST with "1" at 0.47-0.64 even though the prompt asks for a word;
- **`<|im_end|>`** - 0.10-0.21 for the 0.8B, i.e. ending the turn instead of answering;
- **Chinese** - `相似`, `老`, `新`, `红色`, `绿色`;
- **out-of-set colors** - ` gray`, ` grey`, ` brown`, which are not among the six names.

Two A/Bs settled the layout (`configs/archive/scoring_ab.yaml`, `configs/archive/scoring_ab2.yaml`, 50 episodes per
cell, all three models, decision rules fixed before each run):

| Layout | Choice mass | Top-answer share | Accuracy |
|---|---|---|---|
| `words` | 0.13-0.79 | up to 100% (0.8B answers "similar" on every MST trial) | highest |
| `numbered` (mapping reshuffled per trial) | **0.94-1.00** | 20-74% | lowest |
| `fixed` (constant mapping) | 0.94-1.00 | **up to 98%** (2B answers "old" = digit 1) | middling |

`fixed` was introduced to test whether `numbered` loses accuracy because it also asks the model to
hold a per-trial mapping. It does not answer that question, because it failed on bias first: with a
constant mapping the 2B put 88-98% of MST answers on digit 1. That is the ordinal pull the per-trial
shuffle exists to prevent, and it is the same effect the `labeled` layout showed in the 2-AFC prompt
A/B.

`numbered` is pinned in all three model configs. Its accuracy sits below `words` almost everywhere
(4B MST n=5: 0.74 vs 0.94), and that gap is **not** resolved - it is either the cost of holding the
mapping or the mapping step being part of what is measured. Treat `numbered` accuracy as a floor on
memory, and do not compare it to the 2-AFC tasks' accuracy as if the scales were the same.

## MST, color memory and harder foils (0.8B / 2B / 4B only)

These were run for the three small models only; **the 9B and 27B have not been run on any of them**, so
the scaling line above 4B is missing exactly the task that still has headroom (MST). 100 episodes per
cell, `numbered` answer layout (so accuracy is a floor, see above).

**MST** (chance 0.33; LDI = p(similar | lure) - p(similar | foil), averaged over seeds):

| Model | n=1 | 5 | 25 | 100 | 250 | 576 |
|---|---|---|---|---|---|---|
| 0.8B acc (LDI) | 0.22 (-0.08) | 0.27 (-0.12) | 0.32 (+0.01) | 0.28 (-0.02) | – | – |
| 2B acc (LDI) | 0.32 (-0.01) | 0.36 (+0.04) | 0.35 (+0.10) | 0.30 (-0.02) | 0.39 (+0.06) | 0.30 (-0.16) |
| 4B acc (LDI) | 0.75 (+0.35) | 0.76 (+0.29) | 0.62 (0.00) | 0.58 (+0.21) | 0.43 (+0.14) | 0.40 (+0.23) |

MST is the only task that gives the 4B a graded, non-saturated curve, with balanced answers throughout.
The 0.8B and 2B are at chance with no lure discrimination at any size.

**Color memory, named** (6 colors, chance 0.17): 4B 0.86 -> 0.38 from n=1 to 540; 2B 0.45 -> 0.24; 0.8B
0.57 -> 0.19 (n=1 to 100). No collapse onto one color (answer shares 20-38%).

**Color memory, continuous** (mean absolute error; chance 90 degrees): 4B 45 at n=1, then 82-91 from
n=5 up; 0.8B 60 / 67 / 68 at n=1 / 5 / 25 and 88 at n=100; 2B 83-93, at chance throughout. This is the
only free-text-scored measure left in the benchmark; parse failures are 0-16% per cell and reported as a
metric. The 0.8B beating the 4B at n=5-25 is unexplained (see "Not yet tested").

*Parser caveat (port audit, 2026-10-05).* These numbers used a parser that took the *first* 1-3 digit
run in the reply; vision-memory takes the *last* number in [0, 360]. `parse_degrees` now matches
vision-memory. Re-parsing the stored replies changes 188 of 1,600 continuous trials (0.8B 105, 2B 83,
4B 0): the 0.8B/2B answer in paragraphs, where the two rules pick different numbers. The figures above
are therefore parser-dependent for the 0.8B/2B, including the 0.8B-beats-4B comparison; they have not
been re-scored.

**Exemplar and state foils** (Brady2008Exemplar/State, n <= 100; exemplar also on THINGS):

| Model | novel | exemplar | state |
|---|---|---|---|
| 0.8B | 0.41-0.71 | 0.44-0.76 | 0.40-0.52 |
| 2B | 0.62-0.93 | 0.61-0.89 | 0.56-0.70 |
| 4B | 0.99-1.00 | 0.94-1.00 | 0.85-0.98 |

State foils separate the 2B from its exemplar scores; the 4B is near ceiling on all three, so harder foils
do not restore headroom. Several 0.8B cells are flagged `BIAS`.

## Associative inference has no working test format

The ported task runs and produces clean-looking numbers, and they are not memory. Across all three
models and every size from 2 to 500 chains, accuracy sat at 0.43-0.64 with the model answering the
same side on ~100% of trials (the 4B and 2B both answered "2" on 50/50 trials at every size). Choice
mass was ~1.0, so this is not a scoring failure.

A direct-association control settles what it is (`$ZROOT/tmp/control_assoc_inf.py`, job 12915072).
On the same episodes and the same layout it asks two questions: the task's own inference question
(cue A -> choose C, never studied) and a **direct** one (cue A -> choose B, a pair that *was*
studied). Both fail identically:

| Model | n_chains | inference | direct |
|---|---|---|---|
| 2B | 2 / 5 / 25 | 0.60 / 0.60 / 0.80 | 0.50 / 0.55 / 0.55 |
| 4B | 2 / 5 / 25 | 0.60 / 0.60 / 0.80 | 0.50 / 0.50 / 0.55 |

A model that scores 1.00 on recognition at n=500 cannot retrieve a directly studied pair out of two
chains. The test turn is not being read, so the inference step is not what is being measured, and the
accuracies above are the rate at which the correct answer happened to land on the model's preferred
side.

A layout A/B tried three test turns -- `original`, `cue_turn` (the cue in its own turn, so the
question turn holds exactly two images, the shape the working 2-AFC tasks use) and `labeled` -- gated
on the direct control. **None passed.** `cue_turn` behaved identically to `original` (the models still
answered "2" on 18-20 of 20 trials), and `labeled` unstuck the bias without lifting direct retrieval
above chance (its two best cells were 14/20, which a binomial test does not separate from 0.5). So the
three-image test turn was not the cause.

The remaining hypothesis, untested: these models bind an image to *text* but not to another *image*.
Paired associates presents image + "Word: X" and works (0.91-1.00 at n=1 on every model; its large-n
scores depend on the study layout, see below); associative inference presents image + image and fails even on a directly studied pair.

**The task is dropped** (2026-09-27, by decision). Its blocks are commented out in the three model
configs so they cannot silently regenerate ~80 runs of response-bias data, and the original results are
archived at `results/archive_2026-09-26_assoc_inf_broken/`. The wrapper and its three layouts remain in
`bench/tasks/associative_inference_eval.py` if anyone picks it up later.

**Re-ported 2026-10-05 (port audit against vision-memory); not yet evaluated.** The first port differed
from the original in ways that bear on the result above:
- Study pairs were marked with a bare "Pair:"; vision-memory labels them "(AB pair)" / "(BC pair)",
  which also tells the model which link of the chain each pair is. Restored.
- Every episode probed `test_phase[0]`, the first-studied chain (the same first-position problem as
  recognition). Now a random chain per episode.
- Only `pair_type: image` was ever run, although the word variant exists -- the image -> word binding
  the hypothesis above says should work.
The wrapper now also has `probe_types`: `direct_ab` (A -> B) and `direct_bc` (B -> C, image -> word for
`word`) alongside `inference`, on the same episode and chain, which builds the direct control into the
task. `configs/assoc_inf_ab.yaml` (2B/4B, three layouts, gate fixed in its header) is set up but not run.
## Prompt wording changes the results

Tested side by side on all three models, both scored by forced choice (`configs/archive/prompt_ab.yaml`, 50 episodes
per cell, jobs 12819745-50). Two layouts of the same question:

- `original` — vision-memory's wording: question first, then the two unlabeled images.
- `labeled` — "Image 1:"/"Image 2:" before each image, question last.

`labeled` was introduced to stop the paragraph replies and did so, but it pairs "first" with "1" and pushed
answers toward "1": 4B serial order at n=2 fell from **1.00 to 0.62**, the 2B from 0.90 to 0.44. On
recognition the two were within noise for the 4B and mixed for the smaller models. All configs now pin
`prompts: [original]`; `labeled` is kept in the code only so archived results stay identified.

Since the forced choice no longer reads the text, verbosity is no longer a reason to change the prompt.

## Published benchmark scores for comparison

None of the Qwen3.5 model cards report a pure multi-image benchmark (MUIRBench, MMIU, BLINK). The closest
proxies are long-video and long-document benchmarks:

| Benchmark | 0.8B | 2B | 4B | 9B | 27B | 35B-A3B | 122B-A10B |
|---|---|---|---|---|---|---|---|
| LVBench (hour-long video) | 45.1 | 57.1 | 66.4 | 70.0 | 73.6 | 71.4 | 74.4 |
| MLVU (long video) | 65.6 | 76.2 | 82.8 | 84.4 | 85.9 | 85.6 | 87.3 |
| VideoMME (no subtitles) | 57.7 | 69.0 | 76.9 | 78.4 | 82.8 | 82.5 | 83.9 |
| MMLongBench-Doc | 33.6 | 45.4 | 54.2 | 57.7 | 60.2 | 59.5 | 59.0 |

The 0.8B trails the 2B by 10–12 points and the 4B by about 20, and gains flatten from 9B up. Video frames
are temporally related, unlike a set of unrelated object photos, so these are proxies only.


## Scaling above 4B: what the larger models change

Five model sizes have now been run on the three original tasks, 100 episodes per cell. Accuracy on
Brady2008, `novel` foils (the 0.8B's n=250/500 cells were added 2026-09-28 by
`configs/archive/qwen35_0.8b_large.yaml`, 2 seeds x 50 rather than its usual 5 x 20; both are 100 independent
episodes):

| Task | n | 0.8B | 2B | 4B | 9B | 27B |
|---|---|---|---|---|---|---|
| 2-AFC recognition | 1 / 100 / 500 | 0.71 / 0.51 / 0.50 | 0.93 / 0.84 / 0.54 | 0.99 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| Serial order | 2 / 100 / 500 | 0.71 / 0.47 / 0.51 | 0.93 / 0.61 / 0.52 | 1.00 / 0.51 / 0.53 | 0.74 / 0.53 / 0.53 | **1.00 / 0.84 / 0.74** |
| Paired associates | 1 / 100 / 500 | 0.98 / 0.37 / 0.04 | 0.94 / 0.05 / 0.06 | 0.98 / 0.20 / 0.05 | **0.98 / 0.73 / 0.69** | 1.00 / 0.66 / 0.46 |

Three findings, none of them smooth scaling:

- **Recognition saturates at the 4B and stays there.** The 9B and 27B are both 1.00 at every size on
  both datasets. With `novel` foils this task has no headroom above 4B, and `exemplar`/`state` foils do
  not restore it (the 4B is 0.85-1.00 on those too). MST is the only recognition-like task that still
  separates models at the top of the range.
- **Serial order is clearly retained at large n only by the 27B.** It holds 0.84 at n=100 and
  0.72-0.74 at n=500 on both datasets, and this survives the hit1/hit2 split. The smaller models' accuracy
  is ~0.5 by n=100, but that is "always answers 1" rather than a clean chance result (bias-free d' at
  n=100 is 0.5-1.1, not 0), and **the 9B's serial-order numbers do not measure memory at all** under the
  current wording (see the `later` result below). Treat the 9B as unmeasured on this task.
- **Paired associates looked non-monotonic in model size -- but under the `original` layout that is
  mostly a layout effect, not a scale effect** (corrected 2026-09-28; see "Paired associates: the
  study-list layout causes swap errors" below). Under `original` the 9B jumps from the 4B's 0.05 at
  n=500 to 0.69, and the 27B drops back to 0.46 (Brady) and 0.37 (THINGS) while beating the 9B at n<=25.
  The replies are clean (single wordpool words; 98-100% of the 27B's and ~87% of the 9B's come from the
  study list), but almost all of the
  27B's large-n errors are the word of the pair studied *just before* the target (66 of 67 at THINGS
  n=250). Counting those as hits, the 27B is at or above the 9B everywhere. **With an explicit pair
  boundary the ranking is monotone: the 27B beats the 9B at every size** (n=500: 0.69 vs 0.42 on Brady,
  0.87 vs 0.26 on THINGS), so the "9B > 27B" result was the layout, not a memory result.
- **The 0.8B does not hold its paired-associates lead.** It is the best small model at n=25-100, but
  falls to 0.14 / 0.04 (Brady) and 0.19 / 0.07 (THINGS) at n=250 / 500 -- still far above chance (1/n)
  and with almost no neighbouring-word swaps (1 of 138). Its recognition and serial order are at chance
  there, answering "1" on 85-100% of trials.

  Both models are **dense** (no expert or MoE key in either `config.json`; 27B 64 layers x 5120, 9B
  32 x 4096), so active-parameter count is not the explanation either. The cards' MoE language applies
  to the A3B/A10B variants.

Every task keeps the whole study list in the prompt, so what these measure is in-context retrieval,
not storage.

### Serial order accuracy hides an asymmetric response bias

Accuracy near 0.50 on serial order does not mean "no memory" -- for every model it means "always
answers 1". Split by which side was correct (Brady2008, hit1 = accuracy when the answer was 1):

| Model | n=2 | 5 | 25 | 100 | 250 | 500 |
|---|---|---|---|---|---|---|
| 4B hit1 / hit2 | 1.00 / **1.00** | 0.83 / 0.92 | 0.98 / 0.56 | 0.98 / 0.08 | 1.00 / 0.06 | 1.00 / 0.02 |
| 9B hit1 / hit2 | 1.00 / **0.48** | 1.00 / 0.28 | 1.00 / 0.37 | 1.00 / 0.10 | 1.00 / 0.11 | 0.98 / 0.04 |
| 27B hit1 / hit2 | 1.00 / **1.00** | 1.00 / 1.00 | 0.95 / 1.00 | 0.67 / 1.00 | 0.54 / 0.87 | 0.62 / 0.88 |

The 4B and 9B hold hit1 at ~1.00 while hit2 collapses: they answer that the first image shown in the
test pair is the earlier one. The 4B has a genuinely balanced regime at n=2-5 and degrades into the
bias by n=100. **The 9B is already in it at n=2**; the `later` A/B below shows this is how it reads the
question, not an order prior or worse memory. The 27B is the exception (below). Free-text agreement with the forced choice is 83-100%, so this is the
model's answer and not a scoring artefact.

That first->1 pairing is the same effect the `labeled` layout produced for the 2B/4B in the 2026-09-22
prompt A/B, which is why `original` was pinned -- yet the 9B shows it under `original`.

**A layout change does not balance it** (but see the `later` wording below, which shows the bias *is*
the prompt's for the 9B). A layout A/B for the 9B (`configs/archive/serial_order_ab_9b.yaml`,
job 12946926, 50 episodes per cell) found neither layout balances it:

| Brady2008 | n=2 | 5 | 25 | 100 |
|---|---|---|---|---|
| `original` hit1 / hit2 | 1.00 / 0.50 | 1.00 / 0.29 | 1.00 / 0.33 | 1.00 / 0.19 |
| `labeled` hit1 / hit2 | 1.00 / 0.21 | 1.00 / 0.57 | 0.91 / 0.44 | 0.91 / 0.04 |

`labeled` helps at n=5 and hurts at n=2 and n=100, and on THINGS it is equal or worse at every size
(at n=100 it answers "1" on 100% of trials, hit2 = 0.00). The rule fixed before the run required hit2
to rise at small n without hit1 collapsing, which does not happen, so `original` stays pinned. As a
by-product the A/B replicates the main sweep on independently sampled episodes (0.72/0.60/0.64 against
0.74/0.62/0.64 at n=2/5/25).

**The 27B breaks the pattern, and its bias runs the other way.** It is balanced at n=2 (hit1 = hit2 =
1.00) and keeps hit2 at 0.87-1.00 all the way to n=500, while hit1 falls to 0.54-0.62; it answers "1"
on only 32-38% of trials. So its 0.72-0.74 at n=500 is real retention with a mild opposite-direction
bias, not the "always say 1" fallback. That the bias can invert is itself evidence that the smaller
models' "1" preference is a fallback when discrimination fails rather than a property of the task.

**Report hit1/hit2 separately for serial order rather than accuracy alone**, for every model.

**The 9B answers the test layout, not the study order: the `later` wording flips its bias**
(2026-09-28, `configs/archive/serial_order_later_ab.yaml`, job 12962914; same seeds, so the same episodes as the
main sweep). Both earlier prompts ask which image appeared *first* and take "1" or "2", so "first" and
"1" may be read as "the image shown first in the test". `later` asks which appeared *later*, which flips
the correct digit on every trial. Complete, 100 episodes per cell:

| 9B | n=2 | 5 | 25 | 100 |
|---|---|---|---|---|
| Brady `original` ("first"): hit1 / hit2, says "1" | 1.00 / 0.48, 76% | 1.00 / 0.28, 85% | 1.00 / 0.37, 79% | 1.00 / 0.10, 95% |
| Brady `later`: hit1 / hit2, says "1" | 0.04 / 1.00, 2% | 0.19 / 1.00, 10% | 0.14 / 1.00, 8% | 0.08 / 1.00, 4% |
| THINGS `original`: hit1 / hit2, says "1" | 1.00 / 0.46, 73% | 1.00 / 0.14, 93% | 1.00 / 0.12, 94% | 1.00 / 0.10, 95% |
| THINGS `later`: hit1 / hit2, says "1" | 0.32 / 1.00, 19% | 0.16 / 1.00, 8% | 0.04 / 1.00, 2% | 0.00 / 1.00, 0% |

The bias follows the wording at every list length and on both datasets: asked "first" it picks the
first-shown test image, asked "later" the second-shown one, even at n=2 where it should remember
perfectly. THINGS n=2 is less extreme (19% "1") than Brady (2%) but on the same side of the rule. By the rule fixed before the run
(says "1" on <= 30% of trials under `later`), this is a prompt effect: **for the 9B, serial order as
worded measures how it reads the question more than what it remembers**, and a fix needs a question that
does not name a position (e.g. showing one image and asking whether it came in the first or second half,
or asking about a single probe's position). Whether the smaller models' "1" fallback is the same
effect is untested; they are balanced at small n, so for them it is at most a fallback when memory fails.

**Why the bias differs by model, and why size does not remove it.** In a forced choice the model must
answer even with no evidence, and what it falls back on is a property of each separately trained
checkpoint, not of its size. Signal detection separates the two: d' (order discrimination; 0 = none,
~4.7 is the ceiling at 50 trials per side) and c (bias; negative leans "1", positive leans "2"). Brady:

| d' (c) | 0.8B | 2B | 4B | 9B | 27B |
|---|---|---|---|---|---|
| n=2 | 1.7 (-1.1) | 2.9 (-0.2) | 4.7 (0.0) | 2.3 (-1.2) | 4.7 (0.0) |
| n=5 | 1.0 (-1.8) | 0.6 (-0.9) | 2.3 (+0.2) | 1.7 (-1.4) | 4.7 (0.0) |
| n=25 | -0.1 (-2.3) | 0.6 (-0.1) | 2.0 (-0.8) | 1.9 (-1.3) | 4.0 (+0.4) |
| n=100 | 0.7 (-1.9) | 0.5 (+0.1) | 0.5 (-1.6) | 1.1 (-1.8) | 2.8 (+1.0) |

- Bias appears where memory fails: the 4B has c = 0 at n=2 and -1.6 at n=100; the 27B has c = 0 at
  n<=5 and +1.0 at n=100. Better memory means fewer no-evidence trials, so bias shows less -- that is
  the sense in which bigger models look less biased, not a rule that bias shrinks with size.
- The *direction* is per model: 0.8B/4B/9B fall back to "1", the 27B to "2", and the 2B barely leans
  (its misses are close to coin flips).
- Removing bias makes the large-n ordering roughly monotone in size (Brady n=100: 0.7 / 0.5 / 0.5 / 1.1
  / 2.8), so the 27B's lower hit1 at n>=100 is its "2" lean, not worse memory.
- The exception is the 9B, biased at n=2 with d' below the 4B's at n<=5 -- explained by the `later`
  result above. d' is imprecise when one hit rate is at 1.00, as the 9B's is.
- Per-seed hit2 agrees (9B n=2: 12/28 and 12/22; 4B: 28/28 and 22/22), and an earlier A/B replicated
  the 9B on independent episodes, so none of this is seed noise.

Use d' as the headline serial-order measure, with accuracy and hit1/hit2 as supporting views.
**Counterbalanced scoring removes it (2026-10-03, `configs/archive/serial_order_counterbalance_ab.yaml`, jobs
13151793-94).** `counterbalance: true` scores every trial a second time on the same study list with the
two test images swapped, and sums the log-odds for the same image across both orders, so a preference
for a test position cancels. Same seeds as the main sweeps, so the same episodes; the order-shown half
reproduced the saved choice on 1,556 / 1,600 trials (the rest were near-chance trials, 13 exact ties,
moved by bf16 noise on a different node). Accuracy, single-order -> counterbalanced:

| | Brady n=2 | 5 | 25 | 100 | THINGS n=2 | 5 | 25 | 100 |
|---|---|---|---|---|---|---|---|---|
| 4B | 1.00 -> 1.00 | 0.91 -> 0.97 | 0.73 -> 0.88 | 0.51 -> 0.60 | 0.98 -> 1.00 | 0.95 -> 0.99 | 0.76 -> 0.87 | 0.63 -> 0.64 |
| 9B | 0.76 -> **1.00** | 0.65 -> 0.95 | 0.65 -> 0.91 | 0.54 -> 0.62 | 0.62 -> **1.00** | 0.56 -> 0.97 | 0.58 -> 0.86 | 0.54 -> 0.76 |

Both models say "1" on 41-57% of trials once counterbalanced. The pre-set rule (9B >= 0.90 at n=2 on
both datasets) was met, so the 9B's short-list deficit was the position bias, not memory, and
**counterbalanced scoring is the final serial-order measure for every model**
(`configs/serial_order_final_{brady,things256}.yaml`). `frac_consistent` (same image chosen in both
orders) stays in the results as a direct measure of how much each cell answers by position. Every
serial-order result without `counterbalance-True` in its filename is single-order and should be read
through hit1/hit2 or d', not accuracy.

## Paired associates: the study-list layout causes swap errors

Found 2026-09-28 by regenerating the seeded study lists (every regenerated list matched the stored
targets). The `original` layout is `[image 1] Word: W1 [image 2] Word: W2 ...` with nothing between
pairs, so the text right before image k is the *previous* pair's word. Swapped answers (a studied word,
but the wrong one), and how many came from the pair just before the target (offset -1):

| | 9B swaps | of which -1 | 27B swaps | of which -1 |
|---|---|---|---|---|
| THINGS n=5 | 26 | 13 | 2 | 2 |
| THINGS n=25 | 45 | 31 | 17 | 17 |
| THINGS n=100 | 31 | 17 | 41 | 40 |
| THINGS n=250 | 22 | 17 | 67 | 66 |
| THINGS n=500 | 24 | 6 | 63 | 54 |
| Brady n=100 | 22 | 8 | 34 | 33 |
| Brady n=250 | 21 | 8 | 46 | 40 |
| Brady n=500 | 18 | 3 | 52 | 36 |

A random swap would land at -1 about 1/n of the time. The 27B finds the right place in the list and
takes the neighbouring word; counting -1 swaps as hits it scores 91-99 at THINGS n=250-500. The 9B makes
the same error at *small* n, which is why its accuracy was flat (~0.70) across list lengths. Extra 9B
seeds (`configs/archive/qwen35_9b_pam_seeds.yaml`, seeds 2-5) confirm that flat curve is real under this layout:
THINGS 0.58 at n=25 and 0.74 at n=500 over 300 episodes each.

**Layout A/B** (`configs/pam_layout_ab.yaml`): `pair_numbered` puts "Pair k:" before each image. Same
seeds, so the same episodes as the main sweep. Complete, 100 episodes per cell (jobs 12962116-17):

| `original` / `pair_numbered` | n=5 | 25 | 100 | 500 |
|---|---|---|---|---|
| 9B Brady | 0.89 / **1.00** | 0.71 / **0.91** | 0.73 / **0.91** | 0.69 / **0.42** |
| 9B THINGS | 0.74 / **0.98** | 0.53 / **0.92** | 0.64 / 0.65 | 0.74 / **0.26** |
| 27B Brady | 0.96 / **1.00** | 0.79 / **0.99** | 0.66 / **0.96** | 0.46 / **0.69** |
| 27B THINGS | 0.98 / **1.00** | 0.83 / **0.99** | 0.59 / **0.98** | 0.37 / **0.87** |

- The boundary removes the small-n swaps (0 swaps at n=5; 3-4 at n=25, none of them -1) and lifts the
  9B by 11-39 points at n=5-25.
- At n=500 it *lowers* the 9B by 27-50 points. On THINGS the -1 swaps come back (15 of 27 at n=100, 35 of
  57 at n=500; PREV_SWAP flagged at both), so there the boundary stops the neighbouring-word error only
  on short lists. On Brady n=500 it is a different error: 42 swaps, only 8 of them -1.
- **The 27B gains at every size, by up to 50 points** (THINGS n=500: 0.37 -> 0.87). Its -1 swaps largely
  disappear on THINGS (2 of 10 at n=500) and partly remain on Brady (18 of 30 at n=500).
- **Under `pair_numbered` the ordering is monotone in size: the 27B beats the 9B everywhere**, most at
  n=500 (0.69 vs 0.42 Brady, 0.87 vs 0.26 THINGS). This matches the prediction written into the config
  before the run, so the "9B beats 27B" inversion was the layout.
- **PAM results are layout-sensitive at every list length, in both directions.** Do not report the
  `original` PAM curves as a memory measure without that caveat. Why the 9B does *better* on 500 pairs
  than 25 under `original` is unexplained.

**Automatic audit.** PAM trials now record `swap_offset` (study position of the reported word minus the
target's), result metrics include `n_swaps` and `frac_swaps_prev_pair`, and `bench.summarize` flags
**PREV_SWAP** when a cell has >= 10 swaps and more than half are -1 (threshold fixed before the A/B).
Results from before 2026-09-28 have no `swap_offset` and never flag; check those with the list
regeneration instead.

## Stimulus set sizes

- Brady2008Objects has 2,400 images, so a single prompt cannot have more than 2,400 distinct Brady objects.
  Past that, the perception test also drew from THINGS (one image per concept, shrunk to 256 px to match
  Brady's token cost), and says so in its output.
- THINGS has 1,854 concepts and 26,107 images. A concept is an object category ("aardvark", "hammer")
  with at least 12 photos each. Tasks use one photo per concept, so the usable pool is 1,854, not 26,107:
  two photos of one concept in a list would be near-duplicates, which is a different test.
- Serial order's opt-in `exemplars_per_concept: k` (THINGS only) draws k photos per concept and never
  tests a same-concept pair. At k=2 the ceiling is 3,708 (configs use 3,500, ~231k tokens at 256 px).
  It is a separate condition, because the study list contains same-concept photos.
- Recognition's standard episode gives every studied item its own foil (2n images), so it stops at
  1,200 Brady / 927 THINGS. `single_foil` draws n + 1 and reaches 2,399 / 1,853; it is the final
  recognition condition (see "Recognition always tested the first studied image" below).
- If a config asks for more images than a dataset holds, surface it before running rather than clamping
  silently.
- MST has 6 sets x 192 pairs = 1,152. An episode needs as many unstudied pairs as studied ones for its
  foils, so the largest study list is 576.
- Brady2013ColorObjects has 540 images, which is the ceiling for color memory.
- Brady2008Exemplar and Brady2008State hold 100 pairs each (200 files), so `exemplar` and `state` foil
  conditions cannot go past n=100 while `novel` reaches 1,200. Recognition curves across foil types are
  therefore only comparable up to n=100.

### THINGS at 256 px is its own condition

The final THINGS runs use 256 px images (~66 tokens each, like Brady) so the dataset rather than the
context sets the largest list: recognition 1,853, PAM 1,637, serial order 1,854. The rule fixed in
`configs/archive/things256.yaml` before running was that 256 px counts as an extension of the 512 px curves
only if every shared cell (n <= 500, same episodes) is within 10 points. It was not:

- PAM: 4B +0.28 and 9B +0.20 (n=25), 0.8B +0.11.
- Recognition: 0.8B -0.26 (n=5), 2B +0.23 (n=250). This comparison predates the fix in "Recognition always
  tested the first studied image", so it is first-position only.
- Serial order: 2B +0.12, 4B -0.11.
- Within the rule: recognition 4B/9B (<= 0.01), PAM 2B (0.05).

So THINGS-256 curves are reported as a separate condition, not joined to THINGS-512 results. Result
files keep them apart by model name (`qwen3.5-<m>-img256`) in that run.
## Recognition always tested the first studied image (found 2026-10-04)

In the standard 2-AFC recognition episode (`tasks/recognition.py`, `AFCRecognitionTask` without
`single_foil`), `test_phase` is built in study order and the bench scores only `test_phase[0]`. **The
tested item was therefore the first image of the study list on every trial** (40 / 40 checked on CPU, both
datasets). Every recognition result without `single_foil-True` in its filename -- all models, all n, all
foil types, THINGS at 512 and 256 px -- measures memory for list position 1 only. That is why the 4B, 9B
and 27B looked perfect to n=1,200: position 1 is easy. Other tasks shuffle which item is tested and are
not affected.

`single_foil` (added 2026-09-30 to raise the n ceiling) tests one item at a random position, and the
bridge cells exposed the difference (accuracy, `single_foil` vs standard, same n):

| | Brady 500 | Brady 1,000 | THINGS-256 500 | THINGS-256 927 |
|---|---|---|---|---|
| 4B | 0.93 vs 1.00 | 0.78 vs 0.99 | 0.87 vs 1.00 | 0.77 vs 0.97 |
| 9B | -- | -- | 0.98 vs 1.00 | 0.91 vs 1.00 |

So **`single_foil` is the final recognition condition** (`configs/recognition_single_foil_{brady,things256}.yaml`,
n=5 to 2,399 Brady and 1,853 THINGS-256, all five models). At its maximum the 4B is at 0.60 (Brady 2,399)
and 0.67 (THINGS 1,853), and the 9B at 0.81 (THINGS 1,853). The standard-mode results are kept, labelled
as first-position recognition. The recognition prompt itself was not changed: the 0.8B/2B default to "1"
where they are at chance, which does not change the reading.

## Pipeline issues found and fixed

- **Generation ran past the end of the turn.** The checkpoint's `eos_token_id` is 248044, but the chat
  template ends turns with `<|im_end|>` (248046), so the model kept writing the next turn
  (`'yes\nuser\n\nassistant<think>...'`). `bench/models.py` now stops on both tokens.
- **ContinuousRecognitionTask silently relaxes its delay bounds** when `n_images < min_delay + 2`. The runner
  records the parameters actually used in `task_params`.
- **THINGS sampling was not random.** `ThingsDataset(n_categories=k)` took the first k concept folders in
  alphabetical order, so every n=1 episode used *aardvark* and n=5 reused the same five concepts. Fixed in
  `src/stimuli.py` by shuffling with the eval seed. All THINGS results from before 2026-09-22 are void.
- **Scoring OOMed at large n.** `model(**inputs)` returns logits for *every* position: at 250 THINGS images
  that is 64k tokens x 248k vocab, about 30 GiB, and 60 GiB at n=500. Sixteen runs died this way before
  `logits_to_keep=1` was added in `bench/models.py`. `generate()` was never affected, so the capacity numbers
  above still hold.
- **Choice probabilities were tied at exactly 0.5.** Summing both tokenisations of a choice put the shared
  leading-space token (p=0.997) in every choice's set. Only ids that decode to the digit are used now, and
  the prefill ends with a space so the digits carry the mass (0.65/0.35 instead of 3e-4 each).
- **PAM: the larger models named the object instead of recalling the word** ("HOCKEY", "MAGNIFYING GLASS"),
  which capped n=1 at 0.39-0.65. A reworded prompt fixed it (n=1 now 0.91-0.98), but it cost the 0.8B at
  n=100 on Brady (0.63 -> 0.37). Prompt sensitivity differs by model size; one prompt is used for all.
- **The capacity probe cost four times what the real trial cost.** `check_image_capacity` built its
  probe from blank images at `max_image_size` (512 px, 258 tokens) whatever the real stimulus was, so
  probing 2,003 Brady objects (66 tokens each) produced a 517k-token prompt against a 262k context. It
  failed, and the wrapper recorded "model rejected 2003 images" for conditions the capacity probe had
  measured at 30 GB. Four of the 2B's largest associative-inference cells were discarded this way.
  `check_image_capacity(n_images, sample_image=...)` now probes with a stimulus from the condition's
  own dataset.
- **The capacity pre-check was removed from the three original tasks** (2026-09-29). Recognition,
  serial order and paired associates still probed with 512 px blanks (~258 tokens) rather than a real
  stimulus, so for Brady (66 tokens) the probe cost ~4x the real trial: above ~1,000 images it would
  have exceeded the context and skipped conditions that fit. The check only ever caught out-of-memory,
  never context overflow, and the ceilings are now known in advance (see "Largest study list per
  task"), so it was removed rather than fixed; an OOM now fails the run, which the runner records and
  retries, and it frees the GPU cache after a failure. MST, color memory and associative inference keep
  their (fixed) check. Episodes are unchanged: re-running the wrappers reproduced saved trials exactly
  (50/50 in six cells). A consequence for existing results: **`peak_gpu_gb` on Brady cells of these
  three tasks is the probe's peak, not the trial's** -- Brady n=500 records the same peak as THINGS
  n=500. Use THINGS cells or the capacity probes for memory estimates.
- **Skipped runs were never retried.** A skip has `error: None`, so `is_done()` counted it as complete and
  a wrong capacity claim was permanent. It bit twice (the 2B above, then 12 of the 27B's Brady n=250/500
  runs on 2 x L40S that fit on larger cards). Since 2026-09-27 `is_done()` treats a skip as not done,
  because a skip is a hardware-dependent claim, not a fact.
- **`peak_gpu_gb` read device 0 only**, understating a sharded model by about half (26 GB reported for
  the 54.71 GB 27B). It now sums across devices.
- **A file-level check misses runs that were never written.** On 2026-09-27 every run dir was free of
  errors and skips, yet 80 expected runs had no file at all (the archived associative-inference cells,
  still listed in three configs). Before any write-up, expand each config (`expand_runs` + `is_done`)
  and check every expected run exists.
- **Exemplar and state foil pairs were not sampled randomly.** `_get_pairs` took
  `range(min(n, len(ds.pair_paths)))`, i.e. the first n pairs in alphabetical order, so every n=1
  episode used the same pair and every n=5 the same five. The 0.8B scored exactly **0.00** on Brady
  state at n=1 (100/100 trials wrong, perfectly anti-correlated with the target) because it was one
  fixed pair. Only n=100 was unaffected, since it uses all 100 pairs. Same class as the THINGS
  alphabetical sampler. Fixed in `tasks/recognition.py`; 108 result files were voided and rerun.
- **Named color targets cycled from the head of the palette.** `NAMED_COLORS[i % 6]` for i in range(n)
  meant the n=1 target was red in **300 of 300** trials and n=5 never contained purple. The 2B
  "scored" 0.98 at n=1 while answering red on 98% of trials - pure response bias reading as near
  perfect memory. Fixed in `tasks/color_memory.py` by shuffling a repeated palette.
- **Anything outside `configs/qwen35_*.yaml` should be assumed stale until it runs.** Three artifacts
  broke on first use this session, all from interface changes made during the forced-choice rewrite:
  `configs/smoke.yaml` was missing the `prompts` key that became required on 2026-09-22,
  `control_2afc.py` called `build_messages()` without its new `prompt` argument, and
  `run_diag.sbatch` had no `PYTHONPATH` so scripts in `$ZROOT/tmp` could not import `bench`.
## Performance

- A job spends 2–3 minutes starting up: importing torch took 20–120 s from group storage, and loading the
  model 50–85 s. The first generation also compiles the linear-attention GPU code (20–140 s), which is cached
  in `~/.triton` across jobs.
- After that, 0.8B generation is about 0.1–0.5 s per short prompt. So one job should load a model once and run
  many conditions; the runner does this.

## Layer variants

Overriding `layer_types` on a pretrained checkpoint leaves the converted layers randomly initialized, because
their weights do not exist in the checkpoint. An overridden model is not a trained model, and benchmarking it
measures a partly random network. It only becomes meaningful with training on top (for example finetuning a
`-Base` checkpoint). See the README.

## Not yet tested

- MST, color memory and exemplar/state foils on the 9B and 27B.
- All five models beyond n=500 (`configs/archive/qwen35_*_xl.yaml`: Brady 1,000 plus each task's dataset
  limit, THINGS 927). Configured, not yet run.
- Paired associates with `pair_numbered` on the 0.8B-4B, and at n=1 and n=250 for the 9B/27B. The
  main-sweep PAM curves all use `original`.
- The 35B-A3B (`configs/archive/qwen35_35b_a3b.yaml`, PAM only, not yet submitted): a third large model for the
  PAM inversion, and the first MoE (~3B active of 35B), so it separates total from active parameters.
- A serial-order question that does not name a position, which the 9B needs before its order memory
  can be measured; and whether the smaller models' "1" fallback is the same wording effect.
- Why the 9B's `original` PAM accuracy is higher at n=500 than at n=25. (Moot for ranking now that
  `pair_numbered` removes it, but unexplained.)
- Qwen3.5-9B and larger models on the perception test. The 2B was tested on 2026-09-26 and passes
  (85-100% from 1 to 256 images, peak 7.7 GB), so only the 0.8B fails image comparison.
- 3,500+ images on an 80-96 GB GPU; the chunked vision encoder; YaRN beyond 262k tokens.
- Why Brady objects are harder for the 0.8B than THINGS photos. Resolution and token count are ruled
  out; stimulus distinctiveness is the remaining hypothesis and is untested.
- A working test format for associative inference (see above).
- Whether the accuracy gap between the `numbered` and `words` answer layouts is the cost of holding a
  per-trial mapping. `fixed` was meant to answer this and failed on bias first.
- Why the 0.8B beats the 4B on continuous color report at n=5-25 (67-68 degrees of error against
  91/86). It echoes paired associates, where the 0.8B is also the best model at n=25-100.
