#!/usr/bin/env bash
# Source this before TensorFlow runs under WSL when CUDA libraries are provided
# by Python wheels in .venv instead of system packages.

set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CUDA_LIB_PATHS=("/usr/lib/wsl/lib")

NVIDIA_SITE="$(find "$PROJECT_ROOT/.venv/lib" -type d -path "*/site-packages/nvidia" -print -quit 2>/dev/null || true)"
if [[ -n "$NVIDIA_SITE" ]]; then
    while IFS= read -r libdir; do
        CUDA_LIB_PATHS+=("$libdir")
    done < <(find "$NVIDIA_SITE" -mindepth 2 -maxdepth 2 -type d -name lib | sort)
fi

CUDA_LIB_PATH="$(IFS=:; echo "${CUDA_LIB_PATHS[*]}")"
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
    export LD_LIBRARY_PATH="$CUDA_LIB_PATH:$LD_LIBRARY_PATH"
else
    export LD_LIBRARY_PATH="$CUDA_LIB_PATH"
fi

CUDA_NVCC_ROOT="$(find "$PROJECT_ROOT/.venv/lib" -type d -path "*/site-packages/nvidia/cuda_nvcc" -print -quit 2>/dev/null || true)"
if [[ -n "$CUDA_NVCC_ROOT" ]]; then
    export PATH="$CUDA_NVCC_ROOT/bin:$PATH"
    export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CUDA_NVCC_ROOT ${XLA_FLAGS:-}"
fi
