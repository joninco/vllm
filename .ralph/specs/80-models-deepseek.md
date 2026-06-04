# 80 — DeepSeek model wiring (V2 / V4 / MTP)

## Purpose

Re-attach the DeepSeek-V2/V4/MTP model classes to the rebased B12X
attention + MoE + spec-decode surfaces. Includes
`get_max_prefill_buffer_size` plumbing, the attention-config wiring
for B12X sparse-MLA, DSV4 compressed-MLA under DCP, and the
DSV4-Flash virtual TP padding.

## Acceptance Criteria

- `deepseek_v4.attention.DeepseekV4SparseMLAAttention` instantiates
  and binds `max_total_seq_len = get_max_prefill_buffer_size(...)`.
- `deepseek_v2.DeepseekV2Attention` retains the same wiring.
- `deepseek_mtp.DeepseekMTPModel` produces draft tokens compatible
  with the spec-decode cluster's runtime.
- DSV4 padded-TP boot completes (commit `4a3dfa3b2`, on dev branch
  — verify status against main before picking).
- DSV4 compressed-MLA under DCP works (commit `b200c769b` — same
  caveat).

## Constraints

- Do not duplicate `get_max_prefill_buffer_size`. Use the helper
  from `vllm.v1.attention.backends.mla.indexer`.
- Keep `max_total_seq_len` propagation to `SparseAttnIndexer.__init__`
  intact — the sweep reserve from PR #1 depends on it.

## Dependencies

- 50 (sparse-MLA) — indexer must be re-anchored first.
- 60 (spec-decode) — MTP wiring depends on draft runtime.

## Commit Map (verify state against `main`; some may already be

present)

- `266e6b6a0` Proper DSV4 sparse MLA b12x integration
- `29ccc1ceb` GLM 5.1 WIP
- DSV4 Flash + virtual TP + DCP commits (see git log on dev branch
  for the full list; many are already present on `joninco/rebase-source`
  (= seeded fix-branch tip)).

## Conflict Hotspots

- `vllm/model_executor/models/deepseek_v2.py` — heavy upstream
  refactor risk.
- `vllm/models/deepseek_v4/attention.py` — exists only on the fork;
  port to whatever module layout upstream now uses.
- `vllm/model_executor/models/deepseek_mtp.py` — MTP scaffold.

## Validation

```bash
.venv/bin/python -c "
from vllm.models.deepseek_v4.attention import DeepseekV4SparseMLAAttention
print('DSV4 sparse MLA class loads')
"
.venv/bin/python -m pytest tests/models/ -v -x -k 'deepseek' 2>&1 | tail -20
```
