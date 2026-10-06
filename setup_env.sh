#!/bin/bash
# Builds the qwen35-bench Python environment on group storage.
# Home is quota-limited (50G), so the env, uv's cache, and uv's managed Python all live under $ZROOT.
set -euo pipefail

ZROOT="/insomnia001/depts/zgroup/zgroup_burg/zgroup/users/pm3361"
ENV_DIR="$ZROOT/envs/qwen35"
export UV_CACHE_DIR="$ZROOT/uv_cache"
export UV_PYTHON_INSTALL_DIR="$ZROOT/uv_python"
UV="${UV:-$HOME/.local/bin/uv}"
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

# GPU nodes have driver 575 (CUDA <= 12.9), which rules out cu130 torch builds.
# torch 2.10 is the newest version with a prebuilt CUDA 12 causal-conv1d wheel;
# newer torch would require compiling causal-conv1d, and the cluster's CUDA module (12.3) is too old.
TORCH_VERSION="2.10.0"
TORCH_INDEX="https://download.pytorch.org/whl/cu128"
CAUSAL_CONV1D_WHEEL="https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.7.0/causal_conv1d-1.7.0+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"

"$UV" venv --python 3.12 "$ENV_DIR"
export VIRTUAL_ENV="$ENV_DIR"

"$UV" pip install "torch==$TORCH_VERSION" torchvision --index-url "$TORCH_INDEX"
"$UV" pip install "$CAUSAL_CONV1D_WHEEL"
"$UV" pip install -r "$REPO_DIR/requirements.txt"

"$UV" pip freeze > "$REPO_DIR/requirements.lock.txt"
echo "Environment ready: source $ENV_DIR/bin/activate"
