#!/usr/bin/env bash
# Launch Kimi K3 QSRT with one full CUDA graph for each decode batch shape.
set -euo pipefail

K3_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
K3_PYTHON_BIN="${K3_PYTHON_BIN:-${K3_SCRIPT_DIR}/.venv/bin/python}"
K3_MODEL_DIR="${K3_MODEL_DIR:-/data/models/Kimi-K3-QSRT-SQG-XOR-CHEB-T12-3p08-v2-mm-mxfp8-model}"
K3_ENABLE_DSPARK="${K3_ENABLE_DSPARK:-1}"
K3_DSPARK_MODEL_DIR="${K3_DSPARK_MODEL_DIR:-/data/models/Inferact-Kimi-K3-DSpark}"
K3_NUM_SPECULATIVE_TOKENS="${K3_NUM_SPECULATIVE_TOKENS:-7}"
K3_DSPARK_ATTENTION_BACKEND="${K3_DSPARK_ATTENTION_BACKEND:-B12X_MLA}"
K3_LANGUAGE_MODEL_ONLY="${K3_LANGUAGE_MODEL_ONLY:-0}"
K3_ENFORCE_EAGER="${K3_ENFORCE_EAGER:-0}"
K3_KLD_CAPTURE_DIR="${K3_KLD_CAPTURE_DIR:-}"
K3_ENABLE_PREFIX_CACHE="${K3_ENABLE_PREFIX_CACHE:-1}"

if [[ ! -x "${K3_PYTHON_BIN}" ]]; then
  echo "Python interpreter not found or not executable: ${K3_PYTHON_BIN}" >&2
  echo "Create the venv with: uv venv --python 3.12" >&2
  exit 1
fi
if [[ ! -d "${K3_MODEL_DIR}" ]]; then
  echo "QSRT checkpoint directory not found: ${K3_MODEL_DIR}" >&2
  exit 1
fi
if [[ "${K3_ENABLE_DSPARK}" != 0 && "${K3_ENABLE_DSPARK}" != 1 ]]; then
  echo "K3_ENABLE_DSPARK must be 0 or 1." >&2
  exit 2
fi
if [[ "${K3_LANGUAGE_MODEL_ONLY}" != 0 && "${K3_LANGUAGE_MODEL_ONLY}" != 1 ]]; then
  echo "K3_LANGUAGE_MODEL_ONLY must be 0 or 1." >&2
  exit 2
fi
if [[ "${K3_ENFORCE_EAGER}" != 0 && "${K3_ENFORCE_EAGER}" != 1 ]]; then
  echo "K3_ENFORCE_EAGER must be 0 or 1." >&2
  exit 2
fi
if [[ "${K3_ENABLE_PREFIX_CACHE}" != 0 && "${K3_ENABLE_PREFIX_CACHE}" != 1 ]]; then
  echo "K3_ENABLE_PREFIX_CACHE must be 0 or 1." >&2
  exit 2
fi
if [[ "${K3_ENABLE_DSPARK}" == 1 ]]; then
  if [[ ! -f "${K3_DSPARK_MODEL_DIR}/config.json" \
    || ! -f "${K3_DSPARK_MODEL_DIR}/model.safetensors" ]]; then
    echo "Kimi-K3 DSpark checkpoint is incomplete: ${K3_DSPARK_MODEL_DIR}" >&2
    exit 1
  fi
  if [[ "${K3_NUM_SPECULATIVE_TOKENS}" != 5 \
    && "${K3_NUM_SPECULATIVE_TOKENS}" != 7 ]]; then
    echo "Kimi-K3 DSpark supports five or seven speculative tokens." >&2
    exit 2
  fi
  if [[ "${K3_DSPARK_ATTENTION_BACKEND}" != B12X_MLA ]]; then
    echo "Kimi-K3 DSpark is qualified with B12X_MLA." >&2
    exit 2
  fi
fi

# Kimi K3 otherwise defaults to breakable CUDA graphs. This launch profile
# requires B12X MLA, KDA, MoE, and dense linears in one full decode graph.
case "${VLLM_USE_BREAKABLE_CUDAGRAPH:-0}" in
  0|false|False|FALSE|no|No|NO|off|Off|OFF|"")
    export VLLM_USE_BREAKABLE_CUDAGRAPH=0
    ;;
  *)
    echo "This launcher requires full, unbroken decode CUDA graphs." >&2
    echo "Do not set VLLM_USE_BREAKABLE_CUDAGRAPH=1 for this run." >&2
    exit 1
    ;;
esac

for arg in "$@"; do
  case "${arg}" in
    --enforce-eager|--enforce-eager=*|\
    --compilation-config|--compilation-config=*|--compilation-config.*|\
    -cc|-cc=*|-cc.*|\
    --attention-backend|--attention-backend=*|\
    --linear-backend|--linear-backend=*|\
    --moe-backend|--moe-backend=*|\
    --speculative-config|--speculative-config=*|\
    --speculative-model|--speculative-model=*)
      echo "Argument ${arg} would override a launcher-owned runtime option." >&2
      exit 1
      ;;
  esac
done

export PYTHONPATH="${K3_SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export CUTE_DSL_ARCH="${CUTE_DSL_ARCH:-sm_120a}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-32}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-SYS}"
export NCCL_PROTO="${NCCL_PROTO:-LL,LL128,Simple}"
export NCCL_BUFFSIZE="${NCCL_BUFFSIZE:-2097152}"
export NCCL_MAX_NCHANNELS="${NCCL_MAX_NCHANNELS:-8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export VLLM_USE_V2_MODEL_RUNNER="${VLLM_USE_V2_MODEL_RUNNER:-1}"
export VLLM_ENABLE_PCIE_ALLREDUCE="${VLLM_ENABLE_PCIE_ALLREDUCE:-1}"
export VLLM_PCIE_ALLREDUCE_BACKEND="${VLLM_PCIE_ALLREDUCE_BACKEND:-b12x}"
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-1800}"
export VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE="${VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE:-134217728}"

export VLLM_USE_B12X_FP8_GEMM="${VLLM_USE_B12X_FP8_GEMM:-1}"
export VLLM_USE_B12X_MOE="${VLLM_USE_B12X_MOE:-1}"
export B12X_MOE_FORCE_A16="${B12X_MOE_FORCE_A16:-1}"
export KDA_DISABLE_AUTOTUNE="${KDA_DISABLE_AUTOTUNE:-1}"

export INSTANTTENSOR_BACKEND="${INSTANTTENSOR_BACKEND:-AIO}"
export INSTANTTENSOR_MAX_FREE_MEM_USAGE="${INSTANTTENSOR_MAX_FREE_MEM_USAGE:-0.6}"
export SAFETENSORS_FAST_GPU="${SAFETENSORS_FAST_GPU:-1}"

K3_KV_CACHE_ARGS=()
K3_KV_CACHE_MEMORY_BYTES="${K3_KV_CACHE_MEMORY_BYTES:-3865470566}"
if [[ -n "${K3_KV_CACHE_MEMORY_BYTES}" \
  && "${K3_KV_CACHE_MEMORY_BYTES}" != "0" \
  && "${K3_KV_CACHE_MEMORY_BYTES}" != "auto" ]]; then
  K3_KV_CACHE_ARGS+=(
    --kv-cache-memory-bytes "${K3_KV_CACHE_MEMORY_BYTES}"
  )
fi

K3_EXECUTION_ARGS=()
if [[ "${K3_ENFORCE_EAGER}" == 1 ]]; then
  K3_EXECUTION_ARGS+=(--enforce-eager)
else
  # FULL_DECODE_ONLY captures the whole model for uniform single-token decode
  # while leaving prefill outside CUDA graphs. Custom ops remain opaque launches
  # inside the outer graph; they are not eager regions or capture boundaries.
  K3_COMPILATION_CONFIG='{"cudagraph_mode":"FULL_DECODE_ONLY","custom_ops":["all"]}'
  K3_EXECUTION_ARGS+=(--compilation-config "${K3_COMPILATION_CONFIG}")
fi

K3_SPECULATIVE_ARGS=()
if [[ "${K3_ENABLE_DSPARK}" == 1 ]]; then
  printf -v K3_SPECULATIVE_CONFIG \
    '{"method":"dspark","model":"%s","num_speculative_tokens":%s,"attention_backend":"%s","kv_cache_dtype":"fp8","draft_sample_method":"probabilistic","rejection_sample_method":"block"}' \
    "${K3_DSPARK_MODEL_DIR}" "${K3_NUM_SPECULATIVE_TOKENS}" \
    "${K3_DSPARK_ATTENTION_BACKEND}"
  K3_SPECULATIVE_ARGS+=(--speculative-config "${K3_SPECULATIVE_CONFIG}")
fi

K3_LANGUAGE_MODEL_ARGS=()
if [[ "${K3_LANGUAGE_MODEL_ONLY}" == 1 ]]; then
  K3_LANGUAGE_MODEL_ARGS+=(--language-model-only)
fi

K3_PREFIX_CACHE_ARGS=(--no-enable-prefix-caching)
if [[ "${K3_ENABLE_PREFIX_CACHE}" == 1 ]]; then
  K3_PREFIX_CACHE_ARGS=(--enable-prefix-caching)
fi
if [[ -n "${K3_KLD_CAPTURE_DIR}" ]]; then
  export VLLM_KLD_CAPTURE_DIR="${K3_KLD_CAPTURE_DIR}"
  K3_PREFIX_CACHE_ARGS=(--no-enable-prefix-caching)
fi

exec "${K3_PYTHON_BIN}" -m vllm.entrypoints.cli.main serve \
  "${K3_MODEL_DIR}" \
  --served-model-name "${K3_SERVED_MODEL_NAME:-Kimi-K3}" \
  --trust-remote-code \
  "${K3_LANGUAGE_MODEL_ARGS[@]}" \
  --reasoning-parser kimi_k3 \
  --tool-call-parser kimi_k3 \
  --enable-auto-tool-choice \
  --host "${K3_HOST:-0.0.0.0}" \
  --port "${K3_PORT:-8000}" \
  --tensor-parallel-size "${K3_TP_SIZE:-12}" \
  --load-format instanttensor \
  --moe-backend b12x \
  --linear-backend b12x \
  --attention-backend B12X_MLA \
  "${K3_EXECUTION_ARGS[@]}" \
  "${K3_SPECULATIVE_ARGS[@]}" \
  --additional-config '{"kda_prefill_backend":"triton"}' \
  --enable-chunked-prefill \
  "${K3_PREFIX_CACHE_ARGS[@]}" \
  --max-model-len "${K3_MAX_MODEL_LEN:-262144}" \
  --kv-cache-dtype fp8 \
  --block-size "${K3_BLOCK_SIZE:-128}" \
  --gpu-memory-utilization "${K3_GPU_MEMORY_UTILIZATION:-0.9711}" \
  "${K3_KV_CACHE_ARGS[@]}" \
  --max-num-batched-tokens "${K3_MAX_NUM_BATCHED_TOKENS:-1024}" \
  --max-num-seqs "${K3_MAX_NUM_SEQS:-3}" \
  "$@"
