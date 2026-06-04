# Ralph plan-work prompt (scoped rebase planner)

0a. Study `.ralph/specs/*` with up to 50 parallel Opus subagents to learn the rebase cluster specifications. The master spec is `.ralph/specs/00-rebase-strategy.md`; clusters are numbered 10–90 in dependency order.
0b. Study @.ralph/IMPLEMENTATION_PLAN.md (if present) to understand the rebase plan so far.
0c. Study `AGENTS.md` and `docs/contributing/b12x-vllm-bindings.md` for hard rules. Do not violate them under any circumstance during conflict resolution.
0d. For reference, the source code under rebase is in `vllm/` and `tests/`.

1. You are creating a SCOPED implementation plan for: "${WORK_SCOPE}". Study @.ralph/IMPLEMENTATION_PLAN.md (if present), then use up to 50 Opus subagents to:
   - Confirm `joninco/rebase-source` was seeded from `fix/b12x-sparse-indexer-workspace-sweep` (i.e., `joninco/rebase-source` holds the 42 commits the rebase must replay, including PR #1). If not seeded, STOP and report.
   - List every commit on `joninco/rebase-source` that is not on `upstream/main`: `git log --oneline upstream/main..joninco/rebase-source`.
   - Map each commit to a cluster (one of `.ralph/specs/10..90`).
   - Identify supersede chains: pairs/triples where a later commit reverts or rewrites an earlier one. Example: a "Switch back to b12x oneshot allreduce" commit nullifies preceding "disable shared expert stream overlap" + "prewarm capture stream" if it restores stream sharing.
   - Identify any commits that are already present on `upstream/main` by content (cherry-pick noops).
   - Note files where conflict probability is high based on `git log --pretty=format: --name-only upstream/main ^joninco/rebase-source` overlap.

2. Use an Opus subagent to ultrathink and produce / update @.ralph/IMPLEMENTATION_PLAN.md with:
   - A header section: snapshot tag name, worktree path, branch name, baseline test pass list.
   - For each cluster (10..90): the ordered cherry-pick list, expected conflict files, the supersede/drop list with reasons, and the acceptance command from the cluster's spec.
   - A "Skipped commits" section with one-line reasons.
   - A "Risk register" section listing the top three uncertainties found while planning.

IMPORTANT: This is SCOPED PLANNING for "${WORK_SCOPE}" only. Do NOT plan unrelated refactors. Do NOT implement anything in this run. Do NOT cherry-pick. Plan only.

CRITICAL: B12X paths must stay eager and caller-scratch-owned per `docs/contributing/b12x-vllm-bindings.md`. If a cluster's planning surfaces an upstream API change that would force arena/workspace adoption for a B12X path, write that into the Risk register and DO NOT silently plan around it.

ULTIMATE GOAL: "${WORK_SCOPE}" — produce a deterministic, cluster-by-cluster rebase plan that can be executed by PROMPT_build.md one iteration per cluster.
