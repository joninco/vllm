# 90 — Entrypoints: tool/reasoning parsers, score endpoint, vocab pad

## Purpose

Final cluster. Carry over user-facing API additions: tool-call
truncation diagnostics, V2 model runner reasoning parser support,
the `score` endpoint for prompt scoring + KLD output, and the local
top-token vocab padding fix.

## Acceptance Criteria

- `--tool-call-parser deepseek_v4` continues to parse DSV4 tool
  calls; truncation diagnostics fire on malformed inputs.
- `--reasoning-parser deepseek_v4` works under
  `VLLM_USE_V2_MODEL_RUNNER=1`.
- `POST /v1/score` returns prompt scores and KLD output (commit
  `03274ac86`).
- Local top-token vocab padding (commit `b1cab16e6`) — no
  out-of-bounds index errors on TP > 1 boots with vocab-padded
  models.

## Constraints

- API surface must remain backward-compatible with existing clients.
- Score endpoint registers only when explicitly enabled (env or
  config), do not change the default serve behavior.

## Dependencies

- 60 (spec-decode) — reasoning parser interacts with sampler hooks.

## Commit Map

- `540cf4d3f` Add tool call truncation diagnostics
- `0f5c4cc2d` reasoning: allow parsers with V2 model runner
- `03274ac86` score: add prompt scoring and KLD output mode
- `b1cab16e6` Fix local top-token vocab padding handling

## Conflict Hotspots

- `vllm/entrypoints/openai/chat_completion/serving.py` — upstream
  modified the chat-completion serving entry; re-anchor parser
  hooks.
- `vllm/entrypoints/openai/api_server.py` (or split modules
  upstream) — register the score route.

## Validation

```bash
.venv/bin/python -c "
from vllm.entrypoints.openai.tool_parsers import deepseek_v4  # noqa
print('tool parser registers')
"
# Smoke: launch server, curl /v1/score with a small prompt.
```
