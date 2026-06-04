# 20 — Quantization (W4A16, ModelOpt FP8, FlashInfer pin)

## Purpose

Carry forward the W4A16 conversion-overhead fix, ModelOpt mixed
FP8_PB_WO MoE support, FlashInfer fp8 GEMM backend pin, and the
optional "humming" probe path on top of upstream's recent
quantization refactors.

## Acceptance Criteria

- Models that previously loaded W4A16 weights continue to load and
  decode with throughput within 5% of pre-rebase numbers.
- ModelOpt models with mixed FP8_PB_WO MoE layers load without error
  and produce a top-1 token match (greedy) versus the pre-rebase
  reference for the canonical eval prompt.
- The FlashInfer fp8 GEMM backend remains the default when
  `kernel.flashinfer_fp8_gemm=true`; autotune defaults restored.
- "Humming" probing is skipped unless explicitly enabled.
- `tests/quantization/` relevant tests pass.

## Constraints

- Respect upstream's new `vllm/model_executor/layers/quantization/`
  module layout. Do not re-introduce removed symbol re-exports.
- Keep `kernel.flashinfer_fp8_gemm` and related env knobs
  backward-compatible.

## Dependencies

- 10 (build) — needs CUDA 13 build green to import quantization.

## Commit Map

- `28be48fa9` Reduce W4A16 conversion overhead
- `337c8a901` Fix W4A16
- `e61816107` Support ModelOpt mixed FP8_PB_WO MoE
- `4c5fc9667` quantization: make humming probing optional
- `f32636dc4` kernel: pin FlashInfer fp8 GEMM backend
- `5a3da0a07` kernel: restore FlashInfer autotune defaults

## Conflict Hotspots

- `vllm/model_executor/layers/quantization/__init__.py` — upstream
  may have restructured the dispatch table. Re-register W4A16 +
  ModelOpt entries against the new dispatch.
- `vllm/model_executor/layers/quantization/modelopt.py` — upstream
  refactor likely. Re-apply mixed FP8_PB_WO branch on the new
  surface.
- `vllm/model_executor/kernels/linear/__init__.py` — FlashInfer pin
  goes here; ensure it doesn't clobber the upstream default for
  non-pinned cases.
- `vllm/config/kernel.py` — additive options usually merge cleanly.

## Validation

```bash
.venv/bin/python -m pytest tests/quantization/ -v -x \
  --deselect tests/quantization/test_marlin_24.py
.venv/bin/python -c "
from vllm.model_executor.layers.quantization import get_quantization_config
assert get_quantization_config('w4a16') is not None
"
```
