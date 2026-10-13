#!/bin/bash
# Usage: sbatch scripts/bench.sbatch configs/qwen35_0.8b.yaml [--overwrite]
#SBATCH --job-name=qwen35_bench
#SBATCH --partition=short
#SBATCH --account=zgroup
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
set -euo pipefail

CONFIG="${1:?usage: sbatch scripts/bench.sbatch <config.yaml>}"
REPO_DIR="/insomnia001/home/pm3361/qwen35-bench"
ZROOT="/insomnia001/depts/zgroup/zgroup_burg/zgroup/users/pm3361"

export HF_HOME="$ZROOT/hf_cache"
export HF_HUB_OFFLINE=1        # models are pre-downloaded on the login node
export PYTHONUNBUFFERED=1
# Triton's kernel cache defaults to ~/.triton; a full home quota killed a run mid-way (2026-10-09).
export TRITON_CACHE_DIR="$ZROOT/triton_cache"

cd "$REPO_DIR"
source "$ZROOT/envs/qwen35/bin/activate"
python run_bench.py "$CONFIG" "${@:2}"
