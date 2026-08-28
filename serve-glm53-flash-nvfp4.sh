#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${SCRIPT_DIR}/.venv/bin/python}"

MODEL_PATH="${MODEL_PATH:-/data/models/GLM-5.3-Flash-NVFP4}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-zai-org/GLM-5.3-Flash}"
LOAD_FORMAT="${LOAD_FORMAT:-instanttensor}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8001}"
DEFAULT_DEVICE_IDS=8,9
DEVICE_IDS="${DEVICE_IDS:-${DEFAULT_DEVICE_IDS}}"
TP_SIZE="${TP_SIZE:-2}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.94}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-auto}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
SPECULATOR="${SPECULATOR:-mtp}"
DFLASH2_MODEL="${DFLASH2_MODEL:-incoai/GLM-5.3-Flash-DFlash2}"

case "${SPECULATOR}" in
  mtp)
    default_num_speculative_tokens=5
    ;;
  dflash2)
    # The checkpoint is trained for an eight-token block: one verified token
    # plus seven draft tokens.
    default_num_speculative_tokens=7
    ;;
  *)
    echo "SPECULATOR must be mtp or dflash2; got '${SPECULATOR}'" >&2
    exit 2
    ;;
esac

NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-${default_num_speculative_tokens}}"

if [[ ! "${NUM_SPECULATIVE_TOKENS}" =~ ^[0-9]+$ ]]; then
  echo "NUM_SPECULATIVE_TOKENS must be a non-negative integer; got '${NUM_SPECULATIVE_TOKENS}'" >&2
  exit 2
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python interpreter not found or not executable: ${PYTHON_BIN}" >&2
  exit 1
fi
HUMMING_NVRTC_LIB_DIR="$(
  "${PYTHON_BIN}" -c \
    'import sysconfig; print(sysconfig.get_path("purelib") + "/nvidia/cu13/lib")'
)"
if [[ ! -f "${HUMMING_NVRTC_LIB_DIR}/libnvrtc-builtins.so.13.0" ]]; then
  echo "Humming CUDA 13 NVRTC builtins not found: ${HUMMING_NVRTC_LIB_DIR}" >&2
  exit 1
fi
if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "Model config not found: ${MODEL_PATH}/config.json" >&2
  exit 1
fi
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "CUDA_VISIBLE_DEVICES is already set; this launcher selects physical GPUs with --device-ids." >&2
  echo "Run it from an unmasked shell so DEVICE_IDS=${DEVICE_IDS} remains physical." >&2
  exit 2
fi
if [[ ! "${DEVICE_IDS}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "DEVICE_IDS must be a comma-separated list of physical GPU indices; got '${DEVICE_IDS}'" >&2
  exit 2
fi

IFS=, read -r -a device_id_list <<< "${DEVICE_IDS}"
if ((${#device_id_list[@]} != TP_SIZE)); then
  echo "TP_SIZE=${TP_SIZE} requires ${TP_SIZE} DEVICE_IDS; got '${DEVICE_IDS}'" >&2
  exit 2
fi
if [[ "${CUDA_DEVICE_ORDER:-PCI_BUS_ID}" != PCI_BUS_ID ]]; then
  echo "CUDA_DEVICE_ORDER must be PCI_BUS_ID when using physical --device-ids" >&2
  exit 2
fi

export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${HUMMING_NVRTC_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_PLUGINS=
export CUTE_DSL_ARCH="${CUTE_DSL_ARCH:-sm_120a}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-SYS}"
export NCCL_PROTO="${NCCL_PROTO:-LL,LL128,Simple}"
export VLLM_ENABLE_PCIE_ALLREDUCE="${VLLM_ENABLE_PCIE_ALLREDUCE:-1}"
export VLLM_PCIE_ALLREDUCE_BACKEND="${VLLM_PCIE_ALLREDUCE_BACKEND:-b12x}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export SAFETENSORS_FAST_GPU="${SAFETENSORS_FAST_GPU:-1}"
export INSTANTTENSOR_BACKEND="${INSTANTTENSOR_BACKEND:-BUFFERED}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export VLLM_B12X_MOE_FP4_FORCE_A16="${VLLM_B12X_MOE_FP4_FORCE_A16:-1}"

speculative_args=()
if ((NUM_SPECULATIVE_TOKENS > 0)); then
  case "${SPECULATOR}" in
    mtp)
      printf -v speculative_config \
        '{"method":"mtp","num_speculative_tokens":%s,"moe_backend":"humming","attention_backend":"B12X"}' \
        "${NUM_SPECULATIVE_TOKENS}"
      ;;
    dflash2)
      printf -v speculative_config \
        '{"method":"dflash","model":"%s","num_speculative_tokens":%s,"kv_cache_dtype":"auto"}' \
        "${DFLASH2_MODEL}" "${NUM_SPECULATIVE_TOKENS}"
      ;;
  esac
  speculative_args=(--speculative-config "${speculative_config}")
fi

command=(
  "${PYTHON_BIN}" -m vllm.entrypoints.cli.main serve "${MODEL_PATH}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --host "${HOST}"
  --port "${PORT}"
  --device-ids "${DEVICE_IDS}"
  --tensor-parallel-size "${TP_SIZE}"
  --pipeline-parallel-size 1
  --decode-context-parallel-size 1
  --mamba-cache-mode align
  --enable-prefix-caching
  --enable-chunked-prefill
  --dtype bfloat16
  --kv-cache-dtype fp8
  --quantization modelopt_mixed
  --attention-backend B12X
  --block-size 256
  --moe-backend b12x
  --no-enable-flashinfer-autotune
  --load-format "${LOAD_FORMAT}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --max-model-len "${MAX_MODEL_LEN}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
  "${speculative_args[@]}"
  --reasoning-parser glm45
  --tool-call-parser glm47
  --enable-auto-tool-choice
  "$@"
)

cd "${SCRIPT_DIR}"
printf 'Launching %s as %s directly on devices %s\n' \
  "${MODEL_PATH}" "${SERVED_MODEL_NAME}" "${DEVICE_IDS}" >&2
printf 'Serving NVFP4 routed experts through B12X W4A16 (BF16 activations)\n' >&2
printf 'Speculator: %s (%s draft tokens)\n' \
  "${SPECULATOR}" "${NUM_SPECULATIVE_TOKENS}" >&2
exec "${command[@]}"
