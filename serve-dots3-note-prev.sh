#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${SCRIPT_DIR}/.venv/bin/python}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python interpreter not found or not executable: ${PYTHON_BIN}" >&2
  echo "Create the venv with: uv venv --python 3.12" >&2
  exit 1
fi

# The model card's vLLM recipe uses eight GPUs with tensor and expert
# parallelism. Override CUDA_VISIBLE_DEVICES and TP_SIZE together if needed.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"

MODEL_PATH="${MODEL_PATH:-dots-studio/dots3-note-prev-fp8}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-dots3-note-prev}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
TP_SIZE="${TP_SIZE:-8}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"

if ! "${PYTHON_BIN}" - <<'PY'
from vllm.model_executor.models.registry import ModelRegistry

architecture = "Dots3NoteForCausalLM"
raise SystemExit(0 if architecture in ModelRegistry.get_supported_archs() else 1)
PY
then
  echo "This vLLM checkout does not support Dots3NoteForCausalLM." >&2
  echo "Update it to a vLLM revision with native dots3-note support first." >&2
  exit 1
fi

cd "${SCRIPT_DIR}"
exec "${PYTHON_BIN}" -m vllm.entrypoints.cli.main serve "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --enable-expert-parallel \
  --load-format instanttensor \
  --moe-backend deep_gemm \
  --attention-backend B12X_HYBRID_MLA \
  --enable-auto-tool-choice \
  --tool-call-parser dots \
  --reasoning-parser deepseek_v3 \
  --default-chat-template-kwargs '{"enable_thinking": true}' \
  --kv-cache-dtype fp8 \
  --block-size 64 \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  "$@"
