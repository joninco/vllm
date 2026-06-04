# Ralph build prompt (rebase cluster execution)

0a. Study `.ralph/specs/*` with up to 50 parallel Opus subagents to refresh the cluster specifications. Master spec: `.ralph/specs/00-rebase-strategy.md`.
0b. Study @.ralph/IMPLEMENTATION_PLAN.md to find the next unfinished cluster.
0c. Study `AGENTS.md` and `docs/contributing/b12x-vllm-bindings.md`. These are hard rules. Violations abort the run.
0d. Source under rebase is in `vllm/` and `tests/`.

1. Your task is to execute exactly ONE cluster from @.ralph/IMPLEMENTATION_PLAN.md (the next unfinished one, in numerical order). For that cluster:
   a. Confirm you are on branch `ralph/rebase-upstream-catchup` inside the worktree (`../vllm-rebase` or wherever the plan documents). If not, abort and report.
   b. Cherry-pick the commits listed for the cluster in order: `git cherry-pick <hash>`. Skip commits the plan marks as superseded/dropped, citing the reason in the eventual commit message.
   c. On every conflict: read the upstream rewrite of the conflicted file using up to 50 Opus subagents BEFORE editing. Re-apply the B12X / fork hook against the new upstream API surface. NEVER `git checkout --theirs` or `--ours` blindly. NEVER accept "either side" in a hunk without understanding what is being lost.
   d. Run the cluster's acceptance commands listed in the spec. If a command fails, fix the cause; if you cannot, record the failure in `.ralph/IMPLEMENTATION_PLAN.md` under that cluster's section and continue. Do NOT skip the failure silently.
   e. Run validation:
      - `ruff check vllm/ tests/`
      - `.venv/bin/python -c "import vllm"`
      - `pre-commit run --files <files-changed-in-cluster>`
   f. If the cherry-pick produced multiple intermediate commits with fix-ups, squash them WITHIN the cluster only via `git rebase -i <cluster-base>`. Do not rewrite history outside the cluster.
   g. Commit message format:

      ```text
      rebase: <cluster name> onto upstream/main

      <one-line summary of what survived / dropped>

      Co-authored-by: Claude <noreply@anthropic.com>
      ```
   h. `git push joninco ralph/rebase-upstream-catchup`. Do not force-push.

2. After the cluster lands, update @.ralph/IMPLEMENTATION_PLAN.md using a subagent: mark the cluster done, record any deviations, append findings to the Risk register if a deviation matters.

3. STOP. Do not start the next cluster in this same iteration. Each Ralph iteration handles exactly one cluster so the next iteration starts with a fresh context window.

4. Important: B12X paths stay eager and caller-scratch-owned. If you find an upstream API change that would force arena adoption, STOP, document in IMPLEMENTATION_PLAN.md under "Blockers", and exit. Do not bypass the rule.
5. Important: Tests are backpressure. If the cluster's acceptance tests do not exist on `upstream/main`, write them or skip — but do not weaken existing assertions to make a failure pass.
6. As soon as a cluster lands green, the commit doubles as the unit of progress; no need to tag per-cluster.
7. You may add extra logging during conflict resolution but remove it before commit.
8. Keep @.ralph/IMPLEMENTATION_PLAN.md current with cluster status, dropped commits, and any deviation reasons.
9999999999. If @.ralph/IMPLEMENTATION_PLAN.md becomes inconsistent with reality (e.g., a commit is gone), regenerate it via PROMPT_plan_work.md.
99999999999. IMPORTANT: Keep AGENTS.md operational only. Status updates belong in `.ralph/IMPLEMENTATION_PLAN.md`.
999999999999. DO NOT IMPLEMENT PLACEHOLDER OR SIMPLE IMPLEMENTATIONS. Re-anchor every B12X hook against the new upstream API. Stubs waste the next iteration's context.
9999999999999. SUPER IMPORTANT: DO NOT FORCE-PUSH. DO NOT REWRITE HISTORY OUTSIDE THE CURRENT CLUSTER. DO NOT MODIFY THE SNAPSHOT TAG.
