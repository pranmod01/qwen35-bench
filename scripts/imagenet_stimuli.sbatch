#!/bin/bash
# Usage: sbatch scripts/imagenet_stimuli.sbatch [--shards N]
# CPU only: streams ImageNet-21k shards from Hugging Face and saves one 256 px crop per class
# (scripts/build_imagenet_stimuli.py). Resumes from manifest.jsonl if resubmitted.
#SBATCH --job-name=imnet_stim
#SBATCH --partition=short
#SBATCH --account=zgroup
#SBATCH --time=12:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
set -euo pipefail

REPO_DIR="/insomnia001/home/pm3361/qwen35-bench"
ZROOT="/insomnia001/depts/zgroup/zgroup_burg/zgroup/users/pm3361"
OUT="$REPO_DIR/memory_datasets/ImageNet21k_w21_256"

export HF_HOME="$ZROOT/hf_cache"   # the gated-dataset token lives here
export PYTHONUNBUFFERED=1

cd "$REPO_DIR"
source "$ZROOT/envs/qwen35/bin/activate"
python scripts/build_imagenet_stimuli.py --out "$OUT" --targets "$OUT/concepts_fall11_prelim.json" "$@"
