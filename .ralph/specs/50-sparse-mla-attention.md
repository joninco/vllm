# 50 — Sparse MLA / B12X indexer / DCP

## Purpose

The biggest and highest-risk cluster. Re-anchor every B12X sparse
indexer + MLA hook against upstream's restructured
`sparse_attn_indexer.py`, `mla_attention.py`, and indexer metadata
builder. Preserve the workspace-lock fix shipped in
`fix/b12x-sparse-indexer-workspace-sweep` (PR
`local-inference-lab/vllm#1`) and all DCP-aware indexer changes.

## Acceptance Criteria

- `tests/model_executor/layers/test_sparse_attn_indexer_b12x.py
  -k 'extend or profile' -v` — 6/6 pass.
- `tests/v1/attention/test_sparse_mla_backends.py` matches the
  baseline pass/fail profile captured from `joninco/rebase-source`
  (= seeded fix-branch tip).
- B12X attention backend resolves: `--attention-backend
  B12X_MLA_SPARSE` boots without falling back.
- `--enable-chunked-prefill` + B12X sparse indexer survives a
  prefill that crosses the supertile boundary without
  `Workspace is locked but allocation requires X MB` errors.
- `_reserve_b12x_indexer_extend_worst_case` from PR #1 still gets
  invoked at profile time and sweeps both q_cap and q_lo.
- Joint arena preinstall (commits `cd5635a54`, `b98613f27`) runs
  before MoE warmup and before `lock_workspace()`.

## Constraints

- **Hard rule** from `docs/contributing/b12x-vllm-bindings.md`:
  vLLM B12X paths stay eager and caller-scratch-owned. Do NOT
  rewrite scratch reservation through an arena unless upstream
  forces the API contract. If a forced API change appears, document
  the deviation in `.ralph/IMPLEMENTATION_PLAN.md` and pause for
  human review before continuing this cluster.
- Sweep semantics: must visit `q_cap` and
  `q_lo = (nq_max - 1) * BLOCK_Q + 1`. Drop invariant violators
  (`q * k > max_logits_elems` or `k > max_model_len`).
- DCP-aware indexer must still honor `compress_ratio > 1` and the
  active-width window contract from `a125017c9`.

## Dependencies

- 10–40 must be green so `import vllm` succeeds.

## Commit Map (order matters — late commits supersede)

- `1dc84ec5b` attention: fix MLA DCP and fused KV cache handling
- `eca82f1bb` Fix split selection for MLA kernels
- `506557f0d` Reduce B12X MLA workspace reservations
- `fe8c37b59` b12x: support sparse MLA metadata API
- `b098d24ef` Wire B12X sparse indexer override
- `cd5635a54` b12x: preinstall joint arena in V2 runner
- `b98613f27` b12x: preinstall joint arena before MoE warmup
- `76fe76a90` DCP fixes
- (subset of `251eda09f`, `b4943cd56`, `e44b6d5b7` for attention)
- `136d84dd5` sparse indexer: cap k_quant/k_scale workspace at max_model_len
- `356ede945` sparse indexer: sweep chunker envelope to reserve worst-case scratch (PR #1)

## Conflict Hotspots

- `vllm/model_executor/layers/sparse_attn_indexer.py` — already a
  hot file. Upstream may have renamed `current_workspace_manager` or
  the `get_simultaneous` contract. Re-attach
  `_reserve_b12x_indexer_extend_worst_case`,
  `_get_b12x_indexer_extend_buffers`, and the
  `_run_b12x_extend_tiled_topk_streaming` path to the new symbols.
- `vllm/v1/attention/backends/mla/indexer.py` — chunker logic. Keep
  the `chunk_n ≤ workspace_size`, `chunk_m * chunk_n * 4 ≤
  max_logits_bytes` invariants intact; the sweep reservation
  depends on them.
- `vllm/model_executor/layers/attention/mla_attention.py` — DCP +
  split-selection hooks.
- `vllm/v1/worker/workspace.py` — if upstream changed lock
  semantics, re-validate that `_ensure_workspace_size` still rejects
  growth after `lock_workspace()`.
- `vllm/v1/worker/gpu_model_runner.py` — joint arena preinstall
  call site.

## Validation

```bash
.venv/bin/python -m pytest \
  tests/model_executor/layers/test_sparse_attn_indexer_b12x.py \
  -k 'extend or profile' -v
.venv/bin/python -m pytest \
  tests/v1/attention/test_sparse_mla_backends.py -v -x
.venv/bin/python -m pytest \
  tests/v1/attention/test_indexer_deepseek_v4_slot_mapping.py -v
# E2E (hardware-gated):
# bash serve-ds4-flash2.sh &
# curl -X POST localhost:8001/v1/completions -d '{"prompt": "...", "max_tokens": 32}'
# Expect: no "Workspace is locked" line in stderr.
```
