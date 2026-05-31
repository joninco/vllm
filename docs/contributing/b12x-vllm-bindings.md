# B12X vLLM Bindings

Use this guide before touching B12X integration code used from vLLM.

## Ownership Rule

B12X has two separate memory-ownership models:

- **sglang**: a workspace or arena owns memory, is preplanned/cached, and creates the binding with `workspace.bind_*()`.
- **vLLM**: the caller owns memory; code must run eager `plan.bind(scratch=...) -> binding -> kernel`, built fresh per call and CUDA-graph-capturable.

Do not mix these models. In vLLM paths, a binding must never own, cache, construct, or depend on a workspace or arena.

## vLLM Bind Checklist

For vLLM B12X paths:

- Accept caller-provided scratch, typically from vLLM's shared workspace allocation.
- Map that scratch into per-spec views with `narrow`, `view`, or `as_strided`.
- Return a plain views container exposing the attributes kernels read, such as `tmp_output`, `tmp_lse`, `output_buffer`, `num_chunks_ptr`, and `set_split_chunk_config`.
- Do not allocate in `bind()`.
- Do not perform init writes in `bind()`, except guarded scalar control-view fills when required.
- Do not call or construct `B12XAttentionWorkspace`, `_TPCoreArena`, `_make_workspace_views`, `from_shared_arena`, `workspace.bind_*()`, or cached workspace pools from vLLM paths.

Per-call state belongs in the caller, launch wrapper, or kernel prologue, not in a bind-time arena.

## Porting Pattern

Use `B12XCompressedMLAScratch` as the template and `B12XSparseMLAScratch` as the mirror pattern:

1. Define a plain views container.
2. Define layout and materializer from one caller-owned scratch allocation.
3. Rewrite `bind()` to materialize views only.
4. Store views on the binding, not a workspace.
5. Relax type hints to accept duck-typed scratch/views objects where kernels only read attributes.
6. In vLLM, call `plan.bind(scratch=...)` per call; never cache the binding or workspace.
