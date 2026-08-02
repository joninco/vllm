# DeepSeek-V4 compressed-MLA workspace follow-up plan

## Goal

Finish the post-lock workspace fix so the pre-lock reservation covers the full
decode and extend envelope for both DCP and non-DCP execution, while keeping the
exhaustive contract sweep inexpensive and independently tested.

## Production changes

- [x] Make the reserve use the gathered runtime head count:
  `local_padded_heads * decode_context_parallel_size`.
- [x] Compute the 64-wide split cap through one shared helper and pass that exact
  cap into the exhaustive q-chunk sweep.
- [x] Memoize the exhaustive sweep by its complete contract inputs so repeated
  dummy runs and layers with the same geometry reuse the result.
- [x] Keep `max_q_rows=max_num_batched_tokens` and pass the swept maximum through
  `max_q_chunks`; do not grow or unlock WorkspaceManager after lock.
- [x] Consolidate the aligned C128 indexed-width calculation used by the reserve
  and both DeepSeek-V4 metadata builders.
- [x] Document the planner-layout and deterministic-reserve properties that make
  post-lock dummy calls safe.

## Tests

- [x] Pin independently measured q-chunk maxima for C1, C4, and C128 instead of
  recomputing the production result in the assertion.
- [x] Pin exact reserve byte sizes, catching both under-reservation and accidental
  multi-gigabyte over-reservation.
- [x] Add DCP coverage proving the reserve plans for gathered heads and dominates
  the DCP runtime envelope.
- [x] Test that repeated identical envelope queries hit the memoized result.
- [x] Replace the decorative decode/extend mode loop with one explicit shared row
  envelope, since both modes use the same scratch planner.
- [x] Add an optional CPU test against the real planner package used by the
  target branch (`b12x` on Eldritch, `sparkinfer` on Gilded Gnosis).

## Validation

- [x] Run the focused compressed-MLA workspace tests.
- [x] Run the adjacent sparse-indexer tests and record any environment-only skip
  or failure.
- [x] Re-run the exact image's CPU planner envelope for the reporter geometry and
  DCP-scaled heads.
- [x] Run targeted Ruff formatting/checks, mypy, and `git diff --check`.

## Validation results

- Focused workspace tests: 5 passed, 1 optional real-planner test skipped because
  the target package is not installed in the local venv.
- Exact reporter-image planner: DCP1 reserves 530.315430 MiB over a
  530.096680 MiB maximum; DCP2 reserves 1060.627930 MiB over a
  1060.190430 MiB maximum.
- Gilded Gnosis adjacent sparse-indexer tests: 30 passed.
- Ruff, Python 3.10 and 3.12 mypy, Markdown lint, remaining pre-commit hooks, and
  `git diff --check` passed. The attention-backend documentation generator was
  skipped after it exposed an unrelated pre-existing `nvfp4_ds_mla` docs drift.
