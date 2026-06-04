# 70 — Warmup / FlashInfer FP8 autotune

## Purpose

Re-apply the warmup-stage stabilization commits for FlashInfer FP8
autotune bucket coverage, including Kimi runner support, on top of
upstream's runner refactors.

## Acceptance Criteria

- DSV4-Flash boot warmup completes without spurious autotune
  recompilations on every request.
- Kimi runner boots and warms its autotune buckets per commit
  `bd5f6df15`.
- FP8 autotune coverage is at least the union of pre-rebase ranges
  (commit `8b5bac606`).
- No new autotune-related warnings in the first 100 requests.

## Constraints

- Bucket lists are deterministic — keep them sorted and unique
  after the merge.
- Do not regress upstream's default bucket set; the fork adds
  buckets, never removes.

## Dependencies

- 40 (fused-MoE) — autotune fires inside the MoE runner.
- 30 (distributed) — autotune is per-rank; needs TP wiring stable.

## Commit Map

- `b54cdf253` warmup: stabilize FlashInfer FP8 autotune buckets
- `bd5f6df15` warmup: support Kimi runner FlashInfer autotune
- `8b5bac606` warmup: improve FlashInfer FP8 autotune coverage

## Conflict Hotspots

- `vllm/v1/worker/gpu_model_runner.py` — autotune trigger lives
  here; merge bucket additions on top of upstream's new warmup
  scaffold.
- Kimi runner module (path moved upstream — verify before picking).

## Validation

```bash
.venv/bin/python -c "
# Smoke import of runner with autotune helpers
from vllm.v1.worker import gpu_model_runner  # noqa
print('warmup module imports ok')
"
# Hardware-gated: watch logs for repeated 'autotune' lines.
```
