#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3,4,5,6}"
export TP_SIZE="${TP_SIZE:-6}"

virtual_tp_sharding="${VIRTUAL_TP_SHARDING:-b12x-padded}"
attention_head_alignment="${B12X_VIRTUAL_TP_ATTENTION_HEAD_ALIGNMENT:-1}"
moe_intermediate_alignment="${B12X_VIRTUAL_TP_MOE_INTERMEDIATE_ALIGNMENT:-32}"

exec "${SCRIPT_DIR}/serve-glm51.sh" \
  --virtual-tp-sharding "${virtual_tp_sharding}" \
  --b12x-virtual-tp-attention-head-alignment "${attention_head_alignment}" \
  --b12x-virtual-tp-moe-intermediate-alignment "${moe_intermediate_alignment}" \
  "$@"
