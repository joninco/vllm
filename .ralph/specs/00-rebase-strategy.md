# 00 — Rebase Strategy (master spec)

## Purpose

Rebase the 42 commits between `upstream/main` and the working tip
`joninco/fix/b12x-sparse-indexer-workspace-sweep` onto a current
`upstream/main`. Source includes all B12X integration (linear / MoE /
sparse-MLA / mHC / WO + SM_120), the DCP / spec-decode / FlashInfer
extensions, the recent k_quant cap fix (commit `136d84dd5`), and the
sweep-reserve fix (commit `356ede945`, PR
`local-inference-lab/vllm#1`).

NOTE: `origin/main` (= `local-inference-lab/vllm:main`) carries 53
commits the fix-branch does not. Those are NOT in scope for this
rebase. Reconcile separately after the catchup branch lands.

## Acceptance Criteria

- A new branch `ralph/rebase-upstream-catchup` on `joninco/vllm` is
  rooted at `upstream/main` and contains every required local change,
  cluster by cluster (see specs 10–90).
- `git diff upstream/main..joninco/ralph/rebase-upstream-catchup -- specs/`
  is empty for upstream-only files unless a cluster explicitly modifies
  them.
- All cluster acceptance tests pass on the rebased branch.
- The baseline test set from "Phase 3" of the rebase plan is at least
  as green on the rebased branch as on `joninco/rebase-source` (= the seeded
  fork tip).
- A PR is opened from `joninco:ralph/rebase-upstream-catchup` into
  `local-inference-lab/vllm:main` with a rebase summary, the
  dropped-commit list, and a baseline-vs-rebased test diff.

## Constraints

- Honor AGENTS.md and `docs/contributing/b12x-vllm-bindings.md`. B12X
  paths must stay **eager** and **caller-scratch-owned**. Do not adopt
  sglang-style workspaces or arenas during conflict resolution.
- All Python invocations go through `uv` / `.venv/bin/python`. Never
  bare `python3` / `pip`.
- Commits must include `Co-authored-by: Claude <noreply@anthropic.com>`
  attribution per AGENTS.md §2.
- Never force-push to `main`. PR-merge only.
- Do not delete or rewrite the snapshot tag `pre-rebase-snapshot-main`
  while the rebase is in flight.

## Dependencies

This is the parent spec for:

- `10-build-scripts.md`
- `20-quantization.md`
- `30-distributed-pcie-allreduce.md`
- `40-fused-moe-b12x.md`
- `50-sparse-mla-attention.md`
- `60-spec-decode.md`
- `70-warmup-autotune.md`
- `80-models-deepseek.md`
- `90-entrypoints-parsers.md`

Clusters must land in roughly the listed order. Lower-numbered
clusters touch fewer hot files and produce signal before the high-risk
attention/sparse-MLA work.

## Operating Procedure

0. Publish the source tip under a dedicated ref (no force-push, no
   collision with the fork's `main` mirror):
   `git push joninco fix/b12x-sparse-indexer-workspace-sweep:refs/heads/rebase-source`.
   After this step, `joninco/rebase-source` carries every commit the
   rebase needs to replay. `origin/main` is intentionally NOT used
   as the source (it has diverged 53 commits ahead and is out of
   scope); the fork's existing `joninco/main` is left untouched.
1. Snapshot: `git tag pre-rebase-snapshot-main joninco/rebase-source`.
   `git push joninco pre-rebase-snapshot-main`.
2. Worktree: `git worktree add ../vllm-rebase upstream/main`. All
   cherry-picks happen in `../vllm-rebase`.
3. Branch in worktree: `git checkout -b ralph/rebase-upstream-catchup`.
4. Per cluster (`.ralph/specs/NN-*.md`):
   - Resolve dead/superseded commits before picking. See
     `.ralph/IMPLEMENTATION_PLAN.md` for the live drop list.
   - `git cherry-pick <commits>` (one cluster at a time).
   - On conflict: read the new upstream API surface in the conflicted
     file, re-apply the B12X hook against the new code. Do NOT take
     either side blindly.
   - Run that cluster's acceptance tests (see the spec).
   - Run `pre-commit run --files <changed>`.
   - Commit fix-ups inline; squash only at end of cluster.
5. After all clusters: run the full baseline test set and diff against
   the pre-rebase snapshot.
6. Open the PR. Do not merge until human review.

## Escape Hatches

- Abort a stuck cluster: `git cherry-pick --abort` then continue with
  the next cluster. Record the skip in `.ralph/IMPLEMENTATION_PLAN.md`
  with reason; do not silently drop.
- Full rollback: `git switch -` back to `pre-rebase-snapshot-main`,
  drop the worktree, restart from step 1.

## Validation

Commands run after every cluster:

```bash
.venv/bin/python -m pytest \
  tests/model_executor/layers/test_sparse_attn_indexer_b12x.py \
  -k 'extend or profile' -v
.venv/bin/python -c "import vllm"
ruff check vllm/ tests/
```

End-to-end serve smoke (DSV4-Flash TP=4 on SM_120 hardware) is the
final gate before opening the PR.
