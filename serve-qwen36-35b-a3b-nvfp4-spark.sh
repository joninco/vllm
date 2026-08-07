#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${SCRIPT_DIR}/.venv/bin/python}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python interpreter not found or not executable: ${PYTHON_BIN}" >&2
  echo "Create the venv with: uv venv --python 3.12" >&2
  exit 1
fi

cd "${SCRIPT_DIR}"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${CUDA_HOME}/bin:${HOME}/.local/bin:${SCRIPT_DIR}/.venv/bin:${PATH}"
export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export TRITON_PTXAS_PATH="${TRITON_PTXAS_PATH:-${CUDA_HOME}/bin/ptxas}"
export CUTE_DSL_ARCH="${CUTE_DSL_ARCH:-sm_121a}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export SAFETENSORS_FAST_GPU="${SAFETENSORS_FAST_GPU:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export LLM_WORKER_MULTIPROC_METHOD="${LLM_WORKER_MULTIPROC_METHOD:-spawn}"

export VLLM_USE_AOT_COMPILE="${VLLM_USE_AOT_COMPILE:-1}"
export VLLM_USE_BREAKABLE_CUDAGRAPH="${VLLM_USE_BREAKABLE_CUDAGRAPH:-0}"
export VLLM_USE_MEGA_AOT_ARTIFACT="${VLLM_USE_MEGA_AOT_ARTIFACT:-1}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-1}"
export VLLM_USE_V2_MODEL_RUNNER="${VLLM_USE_V2_MODEL_RUNNER:-1}"
export VLLM_USE_B12X_FP8_GEMM="${VLLM_USE_B12X_FP8_GEMM:-1}"
export VLLM_USE_B12X_MOE="${VLLM_USE_B12X_MOE:-1}"
export VLLM_ENABLE_PCIE_ALLREDUCE="${VLLM_ENABLE_PCIE_ALLREDUCE:-1}"
export VLLM_PCIE_ALLREDUCE_BACKEND="${VLLM_PCIE_ALLREDUCE_BACKEND:-b12x}"
export VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE="${VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE:-64KB}"

export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-SYS}"
export NCCL_PROTO="${NCCL_PROTO:-LL,LL128,Simple}"

model_path=${MODEL_PATH:-nvidia/Qwen3.6-35B-A3B-NVFP4}
model_revision=${MODEL_REVISION_OVERRIDE:-491c2f1ea524c639598bf8fa787a93fed5a6fbce}
served_model_name=${SERVED_MODEL_NAME:-Qwen3.6-35B-A3B-NVFP4}
host=${HOST:-0.0.0.0}
port=${PORT:-8000}
gpu_memory_utilization=${GPU_MEMORY_UTILIZATION:-0.40}
max_model_len=${MAX_MODEL_LEN:-262144}
max_num_seqs=${MAX_NUM_SEQS:-4}
max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS:-8192}
max_cudagraph_capture_size=${MAX_CUDAGRAPH_CAPTURE_SIZE:-16}
load_format=${LOAD_FORMAT:-fastsafetensors}
attention_backend=${ATTENTION_BACKEND:-B12X_ATTN}
moe_backend=${MOE_BACKEND:-b12x}
linear_backend=${LINEAR_BACKEND:-b12x}

if ! "${PYTHON_BIN}" - <<'PY'
import importlib
import sys

try:
    blockscaled = importlib.import_module("b12x.gemm.blockscaled")
    tensor_fp8 = importlib.import_module("b12x.gemm.tensor_fp8_linear")
    importlib.import_module("b12x._lib.intrinsics")
    importlib.import_module("b12x.moe.fused_moe")
    importlib.import_module("b12x.attention.paged")
except ImportError as exc:
    print(f"B12X preflight failed: {exc}", file=sys.stderr)
    raise SystemExit(1) from exc

if not blockscaled.is_supported():
    raise SystemExit("B12X preflight failed: native NVFP4 GEMM is unsupported")
if not tensor_fp8.is_supported():
    raise SystemExit("B12X preflight failed: tensor FP8 GEMM is unsupported")
PY
then
  echo "Reinstall the local kernels with:" >&2
  echo "  uv pip install --python .venv/bin/python --no-deps \\" >&2
  echo "    --no-build-isolation --editable ${SCRIPT_DIR}/../b12x" >&2
  exit 1
fi

speculative_args=()
case "${VLLM_ENABLE_MTP:-1}" in
  0|false|no|off|"")
    ;;
  1|true|yes|on)
    num_speculative_tokens=${NUM_SPECULATIVE_TOKENS:-3}
    if [[ ! "${num_speculative_tokens}" =~ ^[1-9][0-9]*$ ]]; then
      echo "NUM_SPECULATIVE_TOKENS must be a positive integer" >&2
      exit 2
    fi
    mtp_moe_backend=${QWEN36_MTP_MOE_BACKEND:-b12x}
    printf -v speculative_config \
      '{"method":"mtp","num_speculative_tokens":%s,"moe_backend":"%s"}' \
      "${num_speculative_tokens}" "${mtp_moe_backend}"
    speculative_args+=(--speculative-config "${speculative_config}")
    ;;
  *)
    echo "VLLM_ENABLE_MTP must be a boolean; got ${VLLM_ENABLE_MTP}" >&2
    exit 2
    ;;
esac

exec "${PYTHON_BIN}" -m vllm.entrypoints.cli.main serve \
  "${model_path}" \
  --revision "${model_revision}" \
  --served-model-name "${served_model_name}" \
  --host "${host}" \
  --port "${port}" \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --kv-cache-dtype fp8 \
  --block-size 128 \
  --load-format "${load_format}" \
  --quantization modelopt_fp4 \
  --attention-backend "${attention_backend}" \
  --moe-backend "${moe_backend}" \
  --linear-backend "${linear_backend}" \
  --gpu-memory-utilization "${gpu_memory_utilization}" \
  --max-model-len "${max_model_len}" \
  --max-num-seqs "${max_num_seqs}" \
  --max-num-batched-tokens "${max_num_batched_tokens}" \
  --max-cudagraph-capture-size "${max_cudagraph_capture_size}" \
  --async-scheduling \
  --no-scheduler-reserve-full-isl \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --no-enable-flashinfer-autotune \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}' \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_xml \
  --enable-auto-tool-choice \
  "${speculative_args[@]}" \
  "$@"
