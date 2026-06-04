# 30 — Distributed: PCIe one-shot allreduce, RTX6K DCP

## Purpose

Re-apply the fork's B12X one-shot allreduce wiring (capture stream
reuse, prewarm, fallback toggle) and the RTX6K PCIe DCP
communicator on top of upstream's parallel_state / cuda_communicator
rewrite.

## Acceptance Criteria

- `VLLM_PCIE_ALLREDUCE_BACKEND=b12x` selects the B12X one-shot path
  on TP>1 boots without crashing during CUDA graph capture.
- `VLLM_ENABLE_PCIE_ALLREDUCE=1` enables the fast path on RTX6000 /
  RTX5090 hosts; a CPU-only fallback is still selectable.
- Capture stream is reused per `eb5a54433` semantics (no per-iter
  stream alloc, no orphaned events).
- TP=4 DSV4-Flash boot completes warmup + graph capture under the
  same memory budget as pre-rebase (within ±100 MB).

## Constraints

- Per AGENTS.md / b12x-vllm-bindings: B12X paths must stay eager and
  caller-scratch-owned. Do not let upstream's new communicator coax
  the B12X one-shot into using its workspace pool.
- Do not regress non-B12X (NCCL, custom_all_reduce) default
  behavior. The B12X path is opt-in.

## Dependencies

- 10 (build) — needs the B12X commit pinned and libs installed.
- 20 (quantization) — autotune defaults live nearby; pick that
  first so cuda_communicator imports don't break.

## Commit Map

- `339ce78a9` distributed: add RTX6K PCIe DCP communication support
- `eb5a54433` b12x: reuse capture stream for oneshot allreduce
- `1701f5d0f` b12x: prewarm oneshot allreduce capture stream
- `9a834c4c5` Switch back to b12x oneshot allreduce
- (subset of `12fb5c6c6` covering allreduce backend wiring)

## Supersede Notes

`907717c83` (disable shared expert stream overlap) is dropped here
if `9a834c4c5` restores stream sharing. Verify by reading commit
content before picking; record outcome in IMPLEMENTATION_PLAN.md.

## Conflict Hotspots

- `vllm/distributed/device_communicators/custom_all_reduce.py` —
  upstream rewrote the dispatch entry. Re-route B12X selection
  through the new `select_backend` function (or whatever the new
  name is) instead of patching the old branch.
- `vllm/distributed/device_communicators/cuda_communicator.py` —
  similar.
- `vllm/distributed/parallel_state.py` — environment-driven backend
  selection; merge additively.
- `vllm/config/vllm.py` — additive config fields.

## Validation

```bash
.venv/bin/python -c "
import os
os.environ['VLLM_PCIE_ALLREDUCE_BACKEND'] = 'b12x'
from vllm.distributed.device_communicators import custom_all_reduce
print('import ok')
"
# Real TP=4 boot smoke gated on hardware availability.
```
