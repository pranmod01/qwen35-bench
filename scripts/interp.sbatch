#!/bin/bash
# Usage: sbatch scripts/interp.sbatch interp.phase0.test_harness [args]   (runs `python -m <module> [args]` in the bench env)
#SBATCH --job-name=interp
#SBATCH --partition=short
#SBATCH --account=zgroup
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --exclude=ins037,ins047,ins058,ins081,ins082,ins091,ins092
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
set -euo pipefail

MODULE="${1:?usage: sbatch scripts/interp.sbatch <python.module> [args]}"
REPO_DIR="/insomnia001/home/pm3361/qwen35-bench"
ZROOT="/insomnia001/depts/zgroup/zgroup_burg/zgroup/users/pm3361"

export HF_HOME="$ZROOT/hf_cache"
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="$ZROOT/triton_cache"

cd "$REPO_DIR"
source "$ZROOT/envs/qwen35/bin/activate"
python -m "$MODULE" "${@:2}"
