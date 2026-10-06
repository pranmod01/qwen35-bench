


# qwen35-bench

Benchmark of the Qwen3.5 model family (and layer-level variants) on visual episodic memory tasks.

## Setup

```bash
./setup_env.sh                     # builds the env on group storage (see the script for version pins)
source /insomnia001/depts/zgroup/zgroup_burg/zgroup/users/pm3361/envs/qwen35/bin/activate
hf download Qwen/Qwen3.5-0.8B      # compute nodes run offline, so fetch models on the login node
```

## Running

Everything about a run lives in one config; `configs/qwen35_0.8b.yaml` is the full sweep and `configs/smoke.yaml`
is a small end-to-end check. The full config is embedded in every result file.

```bash
python run_bench.py configs/smoke.yaml --dry-run   # list runs without loading a model
sbatch scripts/bench.sbatch configs/qwen35_0.8b.yaml     # on a GPU node
```

A run is one (model, task, condition, seed) and writes
`results/<run_name>/<task>/<model>__<condition>__seed<n>.json`, holding every trial's raw response, the
metrics, the config, and provenance (git commit, torch/transformers versions). Existing result files are
skipped, so a job can resume; pass `--overwrite` to redo them.

Each model is loaded once and used for all its runs, because startup (imports plus weight loading from
group storage) costs 2-3 minutes while a warm trial takes well under a second.

## Layer variants

A model entry can override the model's mix of linear and full attention layers:

```yaml
  - name: qwen3.5-0.8b-allfull
    model_id: Qwen/Qwen3.5-0.8B
    layer_variant:
      full_attention_interval: 1   # every layer full attention; 0 = all linear
      # or: layer_types: [linear_attention, full_attention, ...]  (one entry per layer)
```

Stock Qwen3.5-0.8B is 24 layers in a repeating 3 linear (Gated DeltaNet) + 1 full pattern.

**A converted layer has no matching weights in the checkpoint, so transformers initialises it randomly.**
An overridden model is therefore not a trained model, and benchmarking it measures a partly random network.
The override is only meaningful with training on top (for example finetuning a `-Base` checkpoint), so treat
it as the hook for that experiment, not as a result in itself. The runner prints a warning and records the
number of converted layers.

## Adding a task

`bench/tasks/<name>.py` provides `NAME`, `expand_conditions(task_cfg)` and `run(evaluator, condition, seed)`,
and gets registered in `bench/registry.py`. Registered: `afc_recognition`, `serial_order`, `paired_associates`,
`mst`, `color_memory` and `associative_inference`. Continuous recognition and visual haystacks are not
ported (dropped 2026-10-05).

## Provenance

`tasks/`, `src/stimuli.py`, `src/metrics.py`, and `evaluators/{base,qwen_evaluator}.py` were copied from
[csiyer/vision-memory](https://github.com/csiyer/vision-memory) (fork `pranmod01/vision-memory`, branch `vlm-test`, commit `a102f4e`).
They are maintained here independently; upstream changes are not tracked.

## Data

Tasks load stimuli from `memory_datasets/` relative to the working directory, so run from the repo root.
`memory_datasets` is a gitignored symlink to `/insomnia001/depts/zgroup/zgroup_burg/zgroup/users/data/memory_datasets`
on group storage (home is quota-limited). It holds:

- Brady2008 (Objects/Exemplar/State), Brady2013ColorObjects, MST, and the word pool: unzipped from
  `memory_datasets_brady.zip` and `memory_datasets_mst.zip` in the vision-memory repo.
- `THINGS`: a symlink to `/insomnia001/depts/zgroup/zgroup_burg/zgroup/users/data/THINGS`, which holds the full
  THINGS object image set (1854 concepts, 26,107 images, 5.2 G) from the THINGSplus OSF project (jum2f).
  Licensed for research, non-commercial use only, with no redistribution, so never commit it.

The older `Brady2008*` folders directly under `users/data/` are not compatible with the tasks: Exemplar/State
there use original filenames instead of the renamed `<item>1.jpg`/`<item>2.jpg` pairs the tasks expect.
