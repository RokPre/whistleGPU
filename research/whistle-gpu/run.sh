#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "$ROOT/../.." && pwd -P)"
cd "$ROOT"

export TMPDIR="$ROOT/.tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export PIP_CACHE_DIR="$ROOT/.cache/pip"
export HF_HOME="$ROOT/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_ASSETS_CACHE="$HF_HOME/assets"
export XDG_CACHE_HOME="$ROOT/.cache/xdg"
export JAX_COMPILATION_CACHE_DIR="$ROOT/.cache/jax"
export CUDA_CACHE_PATH="$ROOT/.cache/cuda"
export TRITON_CACHE_DIR="$ROOT/.cache/triton"
export TORCH_HOME="$ROOT/.cache/torch"
export PYTHONDONTWRITEBYTECODE=1
export NEEDLE_TELEMETRY=0
export DO_NOT_TRACK=1
export NEEDLE3_LIB_PATH="$ROOT/models/cactus/libneedle.so"
export NEEDLE_WHISTLE_WEIGHTS="$ROOT/models/cactus/whistle.cact"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

for path in .tmp .cache .cache/pip .cache/huggingface .cache/huggingface/hub \
    .cache/huggingface/assets .cache/xdg .cache/jax .cache/cuda .cache/triton \
    .cache/torch .venv checkpoints data models results results/profiles; do
    if [[ -L "$ROOT/$path" ]]; then
        printf 'Refusing symlink that could redirect writes outside the experiment: %s\n' "$ROOT/$path" >&2
        exit 1
    fi
done

mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$HF_HOME" "$XDG_CACHE_HOME" \
    "$JAX_COMPILATION_CACHE_DIR" "$CUDA_CACHE_PATH" "$TRITON_CACHE_DIR" "$TORCH_HOME" \
    "$ROOT/checkpoints" "$ROOT/data/librispeech" "$ROOT/models/cactus" "$ROOT/results"

if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
    python3 -m venv "$ROOT/.venv"
fi
PYTHON="$ROOT/.venv/bin/python"
"$PYTHON" -m pip install --disable-pip-version-check --cache-dir "$PIP_CACHE_DIR" \
    --index-url https://pypi.org/simple \
    --constraint "$ROOT/requirements-gpu.txt" --editable "${REPO_ROOT}[train,gpu]"

ACTION="${1:-all}"
case "$ACTION" in
    prepare)
        "$PYTHON" "$ROOT/experiment.py" prepare
        ;;
    cpu)
        "$PYTHON" "$ROOT/experiment.py" prepare
        "$PYTHON" "$ROOT/experiment.py" cpu
        ;;
    gpu)
        "$PYTHON" "$ROOT/experiment.py" prepare
        "$PYTHON" "$ROOT/experiment.py" gpu
        ;;
    all)
        "$PYTHON" "$ROOT/experiment.py" prepare
        "$PYTHON" "$ROOT/experiment.py" cpu
        "$PYTHON" "$ROOT/experiment.py" gpu
        ;;
    *)
        printf 'Usage: %s [all|prepare|cpu|gpu]\n' "$0" >&2
        exit 2
        ;;
esac
