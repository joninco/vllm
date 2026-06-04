# 40 — Fused MoE: B12X backend, shared expert overlap, MoE aliases

## Purpose

Re-register the B12X MoE backend (FP4 + FP8 paths) inside upstream's
new `fused_moe/runner/` API, re-instate the shared-expert aux output
stream wiring, and keep the mixed-compressed-tensors MoE alias
loader.

## Acceptance Criteria

- `VLLM_USE_B12X_MOE=1` selects the B12X runner end-to-end (config
  resolution → runner dispatch → forward path).
- Shared-expert overlap matches commit `907717c83`'s decision: the
  overlap is **disabled** by default per the documented constraint
  in `4f2d06bf4`. Override via env knob if upstream offers a knob.
- Mixed compressed-tensors MoE checkpoints load via the aliases
  added in `e70e042ce`.
- `VLLM_USE_B12X_FP8_GEMM`, `VLLM_USE_B12X_MHC`,
  `VLLM_USE_B12X_WO_PROJECTION` are still respected at runtime.
- `tests/kernels/moe/test_flashinfer_b12x_moe.py` (if exercised) is
  green or has the same failure profile as on `joninco/rebase-source`
  (= seeded fix-branch tip).

## Constraints

- Per b12x-vllm-bindings: scratch is caller-owned. Do not route B12X
  scratch through the new `fused_moe/runner` arena/workspace
  helpers. If upstream's `Runner` base introduces a `workspace`
  parameter, set it to `None` for B12X and supply caller scratch
  inside the runner.
- Keep the shared-expert overlap constraint comment from `4f2d06bf4`
  in source. Do not silently re-enable the optimization.

## Dependencies

- 10 (build), 20 (quantization), 30 (distributed).

## Commit Map

- `8e46bca85` moe: record shared expert aux output stream
- `4f2d06bf4` b12x: document shared expert overlap constraint
- `907717c83` b12x: disable shared expert stream overlap
- `38ab54ed4` Update b12x MoE force A16 env name
- `e70e042ce` model: load mixed compressed-tensors MoE aliases
- (subset of `12fb5c6c6` and `251eda09f` covering MoE wiring)
- `e44b6d5b7` b12x integration update (MoE portions)

## Conflict Hotspots

- `vllm/model_executor/layers/fused_moe/layer.py` — runner
  registration moved; re-register B12X under the new API.
- `vllm/model_executor/layers/fused_moe/runner/shared_experts.py` —
  upstream rewrote shared experts; re-anchor the aux stream + the
  overlap-disabled toggle.
- `vllm/model_executor/layers/fused_moe/config.py` — additive config
  fields.
- `vllm/model_executor/layers/fused_moe/oracle/nvfp4.py` — adjust if
  upstream added NVFP4 oracle entries that overlap with B12X NVFP4
  padding (commit `f56b09ca5` style work).
- `vllm/model_executor/layers/quantization/utils/flashinfer_fp4_moe.py`
  — keep B12X NVFP4 padding logic.

## Validation

```bash
.venv/bin/python -c "
import os
os.environ['VLLM_USE_B12X_MOE'] = '1'
from vllm.model_executor.layers.fused_moe.layer import FusedMoE
print('B12X MoE registration ok')
"
.venv/bin/python -m pytest tests/kernels/moe/ -v -x \
  -k 'b12x or flashinfer' 2>&1 | tail -20
```
