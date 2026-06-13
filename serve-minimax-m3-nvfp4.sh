#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${SCRIPT_DIR}/.venv/bin/python"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python interpreter not found or not executable: ${PYTHON_BIN}" >&2
  echo "Create the venv with: uv venv --python 3.12" >&2
  exit 1
fi

export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export SAFETENSORS_FAST_GPU=1
export CUTE_DSL_ARCH=sm_120a
export VLLM_USE_B12X_MOE=1
export VLLM_USE_B12X_MINIMAX_M3_MSA=1

M3_PROFILE="${M3_PROFILE:-torch}"
PROFILER_ARGS=()
case "${M3_PROFILE,,}" in
  0|false|no|off|"")
    ;;
  1|true|yes|on|torch)
    M3_PROFILE_DIR="/tmp/vllm-profile/minimax-m3-$(date +%Y%m%d-%H%M%S)"
    mkdir -p "${M3_PROFILE_DIR}"
    export VLLM_RPC_TIMEOUT="${VLLM_RPC_TIMEOUT:-1800000}"
    PROFILER_ARGS+=(
      --profiler-config.profiler=torch
      --profiler-config.torch_profiler_dir="${M3_PROFILE_DIR}"
      --profiler-config.torch_profiler_with_stack=true
      --profiler-config.torch_profiler_record_shapes=false
      --profiler-config.torch_profiler_with_memory=false
      --profiler-config.torch_profiler_with_flops=false
      --profiler-config.torch_profiler_use_gzip=true
      --profiler-config.torch_profiler_dump_cuda_time_total=false
      --profiler-config.ignore_frontend=true
      --profiler-config.delay_iterations=0
      --profiler-config.max_iterations=4
      --profiler-config.warmup_iterations=0
      --profiler-config.active_iterations=5
      --profiler-config.wait_iterations=0
    )
    echo "Torch profiling enabled. Traces will be written under: ${M3_PROFILE_DIR}" >&2
    ;;
  cuda|nsys|nsight)
    export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
    PROFILER_ARGS+=(--profiler-config.profiler=cuda)
    echo "CUDA profiler enabled. Use nsys with --capture-range=cudaProfilerApi and drive /start_profile + /stop_profile." >&2
    ;;
  *)
    echo "ERROR: M3_PROFILE must be one of off, torch, cuda, nsys, or nsight; got '${M3_PROFILE}'" >&2
    exit 1
    ;;
esac

cd "${SCRIPT_DIR}"
exec "${PYTHON_BIN}" -m vllm.entrypoints.cli.main serve /models/MiniMax-M3-NVFP4 \
  --served-model-name MiniMax-M3-NVFP4 \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 131072 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 16 \
  --quantization modelopt_fp4 \
  --kv-cache-dtype fp8_e4m3 \
  --attention-backend TRITON_ATTN \
  --block-size 128 \
  --load-format fastsafetensors \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --reasoning-parser minimax_m3 \
  --enable-auto-tool-choice \
  --tool-call-parser minimax_m3 \
  "${PROFILER_ARGS[@]}" \
  "$@"
