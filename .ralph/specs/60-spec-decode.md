# 60 — Speculative decoding (MTP, EAGLE, DCP, rejection sampling)

## Purpose

Restore the fork's MTP / EAGLE extensions: B12X sparse-MLA decode +
extend fast path under spec, bfloat16 draft MLA KV cache, EAGLE
DCP-aware draft attention, GLM/Kimi MTP runtime overrides, rejection
sampling defaults, and hardened MTP sampling output guarding.

## Acceptance Criteria

- `--speculative-config '{"method":"mtp", ...}'` boots on DSV4-Flash
  TP=4 SM_120 without crashing during warmup or graph capture.
- MTP verify uses the B12X sparse-MLA extend path (commit
  `b24c2c18d`); the MTP decode fast path (commit `484726e8f`) is
  active.
- EAGLE draft attention honors DCP (commit `aaefef443`).
- Constant-draft-positions flag (`bcc490231`) initializes correctly.
- Cached draft logits guard handles padded rows (commit
  `951d9a694`).
- Probabilistic rejection sampling config normalizes legacy values
  (commit `ed69977f8`).
- bfloat16 draft MLA KV cache works without dtype mismatches
  (commit `bcef83900`).

## Constraints

- Upstream may have changed the rejection sampling default API. Use
  the upstream default (`aa89d341b`) where it conflicts with legacy
  fork behavior; only override when functionality requires it.
- Do not break vanilla (non-spec) decode.
- Per AGENTS.md: never skip pre-commit hooks. Spec-decode changes
  often trip mypy.

## Dependencies

- 50 (sparse-MLA) — MTP verify/decode hangs off the sparse-MLA
  attention path.
- 40 (fused-MoE) — MTP draft model uses MoE.

## Commit Map

- `9230570c8` spec-decode: support GLM and Kimi MTP runtime overrides
- `aa89d341b` spec-decode: use upstream rejection sampling defaults
- `ed69977f8` config: normalize legacy probabilistic rejection
- `97947a6f9` spec-decode: restore EAGLE position metadata update
- `bcc490231` spec-decode: initialize constant draft positions flag
- `aaefef443` spec-decode: make EAGLE draft attention DCP-aware
- `bcef83900` spec-decode: support bfloat16 draft MLA KV cache
- `f4904ed66` spec-decode: allow DFlash with multimodal target configs
- `951d9a694` spec-decode: guard cached draft logits for padded rows
- `89da7631e` spec-decode: harden MTP sampling outputs
- `484726e8f` b12x: restore MTP sparse MLA decode fast path
- `b24c2c18d` b12x: keep MTP verify on sparse MLA extend path
- `4ee263c70` Fix GLM MTP sampling and tool delta repair

## Conflict Hotspots

- `vllm/config/speculative.py` — upstream config schema churn.
- `vllm/sampling_params.py` — legacy rejection sampling field
  normalization.
- `vllm/model_executor/models/deepseek_mtp.py` — heavy collision
  zone; rebuild on top of upstream's new MTP scaffold.
- `vllm/v1/spec_decode/` (whole subtree) — re-anchor draft attention
    - position metadata hooks against upstream rewrites.

## Validation

```bash
.venv/bin/python -m pytest tests/v1/spec_decode/ -v -x \
  --deselect tests/v1/spec_decode/test_eagle_correctness.py
.venv/bin/python -c "
from vllm.config.speculative import SpeculativeConfig
cfg = SpeculativeConfig(method='mtp', num_speculative_tokens=2)
print('mtp config ok:', cfg)
"
```
