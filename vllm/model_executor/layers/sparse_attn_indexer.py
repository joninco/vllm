# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Custom Sparse Attention Indexer layers."""

import os
from importlib import import_module
from typing import Any

import torch

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm._aiter_ops import rocm_aiter_ops
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.import_utils import has_deep_gemm
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
)
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerMetadata,
)
from vllm.v1.attention.ops.common import pack_seq_triton, unpack_seq_triton
from vllm.v1.worker.workspace import current_workspace_manager

logger = init_logger(__name__)

RADIX_TOPK_WORKSPACE_SIZE = 1024 * 1024
_B12X_COMPRESSED_INDEX_PAGE_SIZE = 64
_B12X_COMPRESSED_INDEX_HEAD_DIM = 128
_B12X_COMPRESSED_INDEX_SCALE_BYTES = 4
_B12X_COMPRESSED_INDEX_PAGE_WIDTH = _B12X_COMPRESSED_INDEX_PAGE_SIZE * (
    _B12X_COMPRESSED_INDEX_HEAD_DIM + _B12X_COMPRESSED_INDEX_SCALE_BYTES
)
_B12X_EXTEND_TOPK_SUPERTILE_K = int(
    os.getenv("VLLM_B12X_NSA_EXTEND_TOPK_SUPERTILE_K", "32768")
)
_B12X_DECODE_TOPK_SUPERTILE_K = int(
    os.getenv(
        "VLLM_B12X_NSA_DECODE_TOPK_SUPERTILE_K",
        os.getenv("VLLM_B12X_NSA_EXTEND_TOPK_SUPERTILE_K", "32768"),
    )
)
_B12X_INDEXER_EXTEND_BLOCK_Q = 32
_B12X_INDEXER_EXTEND_FALLBACK_BLOCK_K = 256
_B12X_INDEXER_DECODE_BLOCK_Q = 32
_B12X_INDEXER_DECODE_BLOCK_K = 512

# MXFP4 layout: 2 values packed per byte, ue8m0 (1-byte) scale per block of 32.
MXFP4_BLOCK_SIZE = 32


def _ceil_div(x: int, y: int) -> int:
    return (int(x) + int(y) - 1) // int(y)


def _round_up_to_multiple(x: int, y: int) -> int:
    return _ceil_div(x, y) * int(y)


def _reserve_b12x_indexer_extend_worst_case(
    *,
    q_quant: torch.Tensor,
    topk_tokens: int,
    head_dim: int,
    fp8_dtype: torch.dtype,
    max_model_len: int,
) -> None:
    """Sweep the chunker envelope and reserve worst-case extend-prefill scratch.

    The prefill chunker enforces ``q_rows * k_rows <= max_logits_elems`` per
    chunk and ``k_rows <= max_k_rows``. Inside that envelope ``tile_logits``
    scratch grows as ``ceil(q/BLOCK_Q)*BLOCK_Q * min(SUPERTILE_K, k_padded)``
    while the topk arrays scale linearly with ``q``. Sizing once at a single
    ``(q_cap, max_logits_elems // q_cap)`` point is exact only when
    ``q_cap * SUPERTILE_K <= max_logits_elems`` — past that, the q*k budget
    saturates and rounding lets ``(q_cap - BLOCK_Q + 1, k_budget_for_that_q)``
    peak above the q_cap point by up to ~30 MB (one extra k-tile bin per
    q-tile). We sweep both q (q_cap and the same ceil(q/BLOCK_Q) bin's lower
    edge) and k (geometric sweep plus SUPERTILE_K, the q*k=E saturation
    boundary, and max_model_len) and call _get_b12x_indexer_extend_buffers at
    each valid point so the workspace manager keeps the true envelope max
    before lock_workspace().
    """
    q_cap = max(1, int(q_quant.shape[0]))
    max_logits_elems = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024 // 4
    supertile_k = max(
        int(_B12X_EXTEND_TOPK_SUPERTILE_K),
        _B12X_INDEXER_EXTEND_FALLBACK_BLOCK_K,
    )

    nq_max = _ceil_div(q_cap, _B12X_INDEXER_EXTEND_BLOCK_Q)
    q_lo = max(1, (nq_max - 1) * _B12X_INDEXER_EXTEND_BLOCK_Q + 1)
    q_set = sorted({q_cap, q_lo})

    base_ks: set[int] = {supertile_k, int(max_model_len)}
    kk = 1
    while kk <= max_model_len:
        base_ks.add(kk)
        kk *= 2

    for q in q_set:
        if q < 1 or q > q_cap:
            continue
        k_budget = max(1, max_logits_elems // q)
        k_hi = min(int(max_model_len), k_budget)
        ks = set(base_ks)
        ks.add(k_budget)
        ks.add(k_hi)
        for k in sorted(ks):
            if k < 1 or k > max_model_len:
                continue
            if q * k > max_logits_elems:
                continue
            _get_b12x_indexer_extend_buffers(
                q_fp8=q_quant[:q],
                topk_tokens=topk_tokens,
                total_seq_lens=k,
                head_dim=head_dim,
                fp8_dtype=fp8_dtype,
                max_k_rows=max_model_len,
            )


def _b12x_sparse_indexer_requested(enabled: bool | None = None) -> bool:
    if enabled is not None:
        return bool(enabled)

    if envs.VLLM_USE_B12X_SPARSE_INDEXER:
        return True

    from vllm.config import get_current_vllm_config_or_none

    vllm_config = get_current_vllm_config_or_none()
    if vllm_config is None:
        return False

    backend = vllm_config.attention_config.backend
    if isinstance(backend, str):
        return backend == "B12X_MLA_SPARSE"
    return getattr(backend, "name", None) == "B12X_MLA_SPARSE"


def _ensure_b12x_sparse_indexer_supported() -> None:
    if not current_platform.is_cuda():
        raise RuntimeError("B12X sparse indexer/top-k requires CUDA.")
    if not current_platform.is_device_capability_family(120):
        raise RuntimeError(
            "B12X sparse indexer/top-k currently requires an SM120 GPU."
        )


def _use_b12x_sparse_indexer(enabled: bool | None = None) -> bool:
    if not _b12x_sparse_indexer_requested(enabled):
        return False
    _ensure_b12x_sparse_indexer_supported()
    return True


def use_b12x_sparse_indexer(enabled: bool | None = None) -> bool:
    return _use_b12x_sparse_indexer(enabled)


def _gather_workspace_shapes(
    total_seq_lens: int,
    head_dim: int,
    fp8_dtype: torch.dtype,
    use_fp4_cache: bool,
) -> tuple[tuple[tuple[int, int], torch.dtype], tuple[tuple[int, int], torch.dtype]]:
    """Return ((values_shape, values_dtype), (scales_shape, scales_dtype)) for
    the K-gather workspace. FP8 path: (T, head_dim) fp8 + (T, 4) uint8 fp32
    scales. MXFP4 path: (T, head_dim // 2) uint8 packed mxfp4 +
    (T, head_dim // MXFP4_BLOCK_SIZE) uint8 ue8m0 scales."""
    if use_fp4_cache:
        return (
            ((total_seq_lens, head_dim // 2), torch.uint8),
            ((total_seq_lens, head_dim // MXFP4_BLOCK_SIZE), torch.uint8),
        )
    return (
        ((total_seq_lens, head_dim), fp8_dtype),
        ((total_seq_lens, 4), torch.uint8),
    )


def kv_cache_as_quant_view(
    kv_cache: torch.Tensor,
    head_dim: int,
    use_fp4_cache: bool,
) -> torch.Tensor:
    """4D ``[num_blocks, block_size, 1, head_width]`` view expected by
    DeepGEMM, from the 3D indexer kv-cache allocation."""
    if use_fp4_cache:
        assert kv_cache.ndim == 3 and kv_cache.dtype == torch.uint8
        num_blocks, block_size, _ = kv_cache.shape
        page_bytes = int(kv_cache.stride(0))
        fp4_bytes = head_dim // 2 + head_dim // MXFP4_BLOCK_SIZE
        return torch.as_strided(
            kv_cache,
            size=(num_blocks, block_size, 1, fp4_bytes),
            stride=(page_bytes, fp4_bytes, fp4_bytes, 1),
        )
    return kv_cache.unsqueeze(-2)


def _get_b12x_indexer_extend_buffers(
    *,
    q_fp8: torch.Tensor,
    topk_tokens: int,
    total_seq_lens: int,
    head_dim: int,
    fp8_dtype: torch.dtype,
    max_k_rows: int | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    from b12x.attention.indexer import resolve_extend_prefill_block_k

    q_rows = max(1, int(q_fp8.shape[0]))
    k_rows = max(1, int(total_seq_lens))
    # Size k_quant/k_scale from the constant capacity cap so the workspace stays
    # constant after locking. The gather kernel still runs over [:k_rows].
    k_rows_cap = max(k_rows, int(max_k_rows)) if max_k_rows is not None else k_rows
    indexer_num_q_heads = int(q_fp8.shape[1])
    prefill_block_k = resolve_extend_prefill_block_k(
        valid_q_rows=q_rows,
        k_rows=k_rows,
        num_heads=indexer_num_q_heads,
    )
    if prefill_block_k is None:
        prefill_block_k = _B12X_INDEXER_EXTEND_FALLBACK_BLOCK_K
    prefill_block_k = int(prefill_block_k)

    num_q_tiles = _ceil_div(q_rows, _B12X_INDEXER_EXTEND_BLOCK_Q)
    num_k_tiles = _ceil_div(k_rows, prefill_block_k)
    supertile_k = _round_up_to_multiple(
        max(int(_B12X_EXTEND_TOPK_SUPERTILE_K), prefill_block_k),
        prefill_block_k,
    )
    supertile_tiles = max(1, supertile_k // prefill_block_k)
    max_chunk_tiles = min(supertile_tiles, num_k_tiles)
    tile_elements = max(
        1,
        num_q_tiles
        * max_chunk_tiles
        * _B12X_INDEXER_EXTEND_BLOCK_Q
        * prefill_block_k,
    )
    topk_tokens = max(int(topk_tokens), 1)

    values_spec, scales_spec = _gather_workspace_shapes(
        k_rows_cap, head_dim, fp8_dtype, use_fp4_cache=False
    )
    (
        k_quant,
        k_scale,
        tile_logits,
        lengths,
        topk_values,
        topk_indices,
        candidate_values,
        candidate_indices,
        merge_positions,
    ) = current_workspace_manager().get_simultaneous(
        values_spec,
        scales_spec,
        ((tile_elements,), torch.float32),
        ((q_rows,), torch.int32),
        ((q_rows, topk_tokens), torch.float32),
        ((q_rows, topk_tokens), torch.int32),
        ((2, q_rows, topk_tokens), torch.float32),
        ((2, q_rows, topk_tokens), torch.int32),
        ((q_rows, topk_tokens), torch.int64),
    )
    return (
        k_quant[:k_rows],
        k_scale[:k_rows],
        tile_logits[:tile_elements],
        lengths[:q_rows],
        topk_values[:q_rows, :topk_tokens],
        topk_indices[:q_rows, :topk_tokens],
        candidate_values[:, :q_rows, :topk_tokens],
        candidate_indices[:, :q_rows, :topk_tokens],
        merge_positions[:q_rows, :topk_tokens],
    )


def _normalize_b12x_indexer_weights(
    weights: torch.Tensor,
    *,
    q_rows: int,
    num_heads: int,
) -> torch.Tensor:
    if weights.ndim == 3:
        if int(weights.shape[2]) != 1:
            raise RuntimeError(
                "b12x extend indexer expected rank-3 weights to have "
                f"trailing dimension 1, got {tuple(weights.shape)}."
            )
        weights = weights.squeeze(2)
    if weights.ndim != 2:
        raise RuntimeError(
            "b12x extend indexer expected weights rank 2 or 3, "
            f"got {weights.ndim}."
        )
    if weights.shape != (q_rows, num_heads):
        raise RuntimeError(
            "b12x extend indexer expected weights shape "
            f"{(q_rows, num_heads)}, got {tuple(weights.shape)}."
        )
    return weights.to(torch.float32)


@triton.jit
def _normalize_prefill_topk_to_req_relative_kernel(
    topk_indices_ptr,
    cu_seq_lens_ptr,
    token_to_seq_ptr,
    topk_cols,
    num_elems,
    topk_stride_0,
    topk_stride_1,
    token_to_seq_len,
    num_reqs,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elems
    rows = offsets // topk_cols
    cols = offsets - rows * topk_cols
    topk_ptrs = topk_indices_ptr + rows * topk_stride_0 + cols * topk_stride_1

    packed_indices = tl.load(topk_ptrs, mask=mask, other=-1)
    valid_indices = mask & (packed_indices >= 0) & (
        packed_indices < token_to_seq_len
    )
    seq_ids = tl.load(
        token_to_seq_ptr + packed_indices,
        mask=valid_indices,
        other=0,
    )
    valid_seq_ids = valid_indices & (seq_ids >= 0) & (seq_ids < num_reqs)
    seq_starts = tl.load(
        cu_seq_lens_ptr + seq_ids,
        mask=valid_seq_ids,
        other=0,
    )
    tl.store(topk_ptrs, packed_indices - seq_starts, mask=valid_seq_ids)


def _normalize_prefill_topk_to_req_relative(
    chunk: object, topk_indices: torch.Tensor
) -> None:
    """Convert packed prefill offsets to per-request token offsets."""
    cu_seq_lens = getattr(chunk, "cu_seq_lens", None)
    token_to_seq = getattr(chunk, "token_to_seq", None)
    if (
        cu_seq_lens is None
        or token_to_seq is None
        or cu_seq_lens.numel() <= 2
        or token_to_seq.numel() == 0
        or topk_indices.numel() == 0
    ):
        return

    if topk_indices.is_cuda:
        block_size = 1024
        grid = (triton.cdiv(topk_indices.numel(), block_size),)
        _normalize_prefill_topk_to_req_relative_kernel[grid](
            topk_indices,
            cu_seq_lens,
            token_to_seq,
            topk_indices.shape[1],
            topk_indices.numel(),
            topk_indices.stride(0),
            topk_indices.stride(1),
            token_to_seq.numel(),
            cu_seq_lens.numel() - 1,
            BLOCK_SIZE=block_size,
        )
        return

    valid = topk_indices >= 0
    safe_indices = topk_indices.clamp(min=0, max=int(token_to_seq.numel()) - 1)
    seq_ids = token_to_seq[safe_indices]
    seq_starts = cu_seq_lens[seq_ids]
    normalized = topk_indices - seq_starts
    topk_indices.copy_(torch.where(valid, normalized, topk_indices))


def _run_b12x_extend_tiled_topk_streaming(
    *,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    kv_fp8: tuple[torch.Tensor, torch.Tensor],
    metadata: Any,
    topk: int,
    contract_phantoms: dict[str, torch.Tensor] | None,
    workspace: Any,
    tile_logits: torch.Tensor,
    lengths: torch.Tensor,
    output_values: torch.Tensor,
    output_indices: torch.Tensor,
    candidate_values: torch.Tensor,
    candidate_indices: torch.Tensor,
    merge_positions: torch.Tensor,
    supertile_k: int,
) -> torch.Tensor:
    from b12x.attention.indexer import resolve_extend_prefill_block_k
    from b12x.attention.indexer.extend_kernel import (
        run_extend_logits_kernel,
        supports_extend_logits_kernel,
    )
    from b12x.attention.indexer.tiled_topk import (
        merge_tiled_topk_candidates,
        run_tiled_topk,
    )

    if q_fp8.ndim != 3:
        raise RuntimeError(
            "b12x extend indexer expected q_fp8 rank 3, "
            f"got {q_fp8.ndim}."
        )
    k_start = metadata.k_start
    k_end = metadata.k_end
    if k_start.ndim != 1 or k_end.ndim != 1 or k_start.shape != k_end.shape:
        raise RuntimeError(
            "b12x extend indexer requires matching rank-1 k_start/k_end, "
            f"got {tuple(k_start.shape)} and {tuple(k_end.shape)}."
        )

    topk = int(topk)
    num_q_rows = int(k_start.shape[0])
    num_heads = int(q_fp8.shape[1])
    weights_f = _normalize_b12x_indexer_weights(
        weights, q_rows=int(q_fp8.shape[0]), num_heads=num_heads
    )
    k_quant, k_scale = kv_fp8

    if not supports_extend_logits_kernel(
        q_fp8=q_fp8,
        weights=weights_f,
        k_quant=k_quant,
        k_scale=k_scale,
        k_start=k_start,
        k_end=k_end,
    ):
        from b12x.attention.indexer.api import _reference_topk_indices_from_logits
        from b12x.attention.indexer.reference import extend_logits_reference

        torch.sub(k_end, k_start, out=lengths[:num_q_rows])
        logits = extend_logits_reference(
            q_fp8=q_fp8,
            weights=weights_f,
            kv_fp8=kv_fp8,
            k_start=k_start,
            k_end=k_end,
        )
        return _reference_topk_indices_from_logits(
            logits[:num_q_rows],
            topk=topk,
            output_values=output_values,
            output_indices=output_indices,
        )

    prefill_block_k = resolve_extend_prefill_block_k(
        valid_q_rows=num_q_rows,
        k_rows=int(k_quant.shape[0]),
        num_heads=num_heads,
    )
    if prefill_block_k is None:
        prefill_block_k = _B12X_INDEXER_EXTEND_FALLBACK_BLOCK_K
    prefill_block_k = int(prefill_block_k)
    block_q = _B12X_INDEXER_EXTEND_BLOCK_Q
    num_q_tiles = _ceil_div(num_q_rows, block_q)
    num_k_tiles = _ceil_div(int(k_quant.shape[0]), prefill_block_k)
    tile_size = block_q * prefill_block_k
    resolved_supertile_k = _round_up_to_multiple(
        max(int(supertile_k), prefill_block_k),
        prefill_block_k,
    )
    supertile_tiles = max(1, resolved_supertile_k // prefill_block_k)
    num_chunks = _ceil_div(num_k_tiles, supertile_tiles)
    max_chunk_tiles = min(supertile_tiles, num_k_tiles)
    chunk_tile_elements = num_q_tiles * max_chunk_tiles * tile_size
    if int(tile_logits.numel()) < chunk_tile_elements:
        raise RuntimeError(
            "b12x extend indexer tile_logits scratch is too small: "
            f"{int(tile_logits.numel())} elements for {chunk_tile_elements}."
        )

    if lengths.ndim != 1 or int(lengths.shape[0]) < num_q_rows:
        raise RuntimeError(
            "b12x extend indexer lengths scratch has shape "
            f"{tuple(lengths.shape)}, expected at least ({num_q_rows},)."
        )
    global_lengths = lengths[:num_q_rows]
    torch.sub(k_end, k_start, out=global_lengths)

    if num_chunks <= 1:
        tile_logits[:chunk_tile_elements].fill_(float("-inf"))
        run_extend_logits_kernel(
            q_fp8=q_fp8,
            weights=weights_f,
            k_quant=k_quant,
            k_scale=k_scale,
            k_start=k_start,
            k_end=k_end,
            contract_phantoms=contract_phantoms,
            workspace=workspace,
            preinitialize_invalid_logits=True,
            tile_logits=tile_logits,
            tile_k_offset=0,
            tile_num_k_tiles=num_k_tiles,
        )
        _, topk_indices = run_tiled_topk(
            tile_logits=tile_logits,
            k_start=k_start,
            lengths=global_lengths,
            topk=topk,
            block_q=block_q,
            block_k=prefill_block_k,
            output_values=output_values,
            output_indices=output_indices,
            num_k_tiles=num_k_tiles,
            contract_phantoms=contract_phantoms,
        )
        return topk_indices

    expected_candidate_shape = (2, num_q_rows, topk)
    if (
        tuple(candidate_values.shape) != expected_candidate_shape
        or tuple(candidate_indices.shape) != expected_candidate_shape
    ):
        raise RuntimeError(
            "b12x extend indexer streaming candidates must have fixed shape "
            f"{expected_candidate_shape}, got {tuple(candidate_values.shape)} "
            f"and {tuple(candidate_indices.shape)}."
        )

    for chunk_idx in range(num_chunks):
        chunk_tile_begin = chunk_idx * supertile_tiles
        chunk_tile_end = min(chunk_tile_begin + supertile_tiles, num_k_tiles)
        chunk_tiles = chunk_tile_end - chunk_tile_begin
        chunk_start = chunk_tile_begin * prefill_block_k
        chunk_rows = chunk_tiles * prefill_block_k
        chunk_elements = num_q_tiles * chunk_tiles * tile_size

        tile_logits[:chunk_elements].fill_(float("-inf"))
        run_extend_logits_kernel(
            q_fp8=q_fp8,
            weights=weights_f,
            k_quant=k_quant,
            k_scale=k_scale,
            k_start=k_start,
            k_end=k_end,
            contract_phantoms=contract_phantoms,
            workspace=workspace,
            preinitialize_invalid_logits=True,
            tile_logits=tile_logits,
            tile_k_offset=chunk_tile_begin,
            tile_num_k_tiles=chunk_tiles,
        )
        candidate_slot = 0 if chunk_idx == 0 else 1
        run_tiled_topk(
            tile_logits=tile_logits,
            k_start=k_start,
            lengths=global_lengths,
            topk=topk,
            block_q=block_q,
            block_k=prefill_block_k,
            output_values=candidate_values[candidate_slot],
            output_indices=candidate_indices[candidate_slot],
            num_k_tiles=chunk_tiles,
            input_index_offset=chunk_start,
            input_extent=chunk_rows,
            output_index_offset=chunk_start,
            contract_phantoms=contract_phantoms,
        )
        if chunk_idx == 0:
            continue

        merge_tiled_topk_candidates(
            candidate_values=candidate_values,
            candidate_indices=candidate_indices,
            topk=topk,
            output_values=output_values,
            output_indices=output_indices,
            merge_positions=merge_positions,
        )
        candidate_values[0].copy_(output_values)
        candidate_indices[0].copy_(output_indices)

    return output_indices


def _flatten_b12x_paged_index_cache(kv_cache: torch.Tensor) -> torch.Tensor:
    expected_shape_tail = (
        _B12X_COMPRESSED_INDEX_PAGE_SIZE,
        _B12X_COMPRESSED_INDEX_HEAD_DIM + _B12X_COMPRESSED_INDEX_SCALE_BYTES,
    )

    if kv_cache.ndim != 3 or kv_cache.dtype != torch.uint8:
        raise RuntimeError(
            "b12x paged indexer cache must be rank-3 uint8 with "
            f"shape [num_blocks, {expected_shape_tail[0]}, "
            f"{expected_shape_tail[1]}], got shape={tuple(kv_cache.shape)} "
            f"dtype={kv_cache.dtype}."
        )
    if tuple(kv_cache.shape[1:]) != expected_shape_tail:
        raise RuntimeError(
            "b12x paged indexer cache has an unsupported shape, "
            f"got {tuple(kv_cache.shape)}; expected tail {expected_shape_tail}."
        )
    if kv_cache.stride(1) != expected_shape_tail[1] or kv_cache.stride(2) != 1:
        raise RuntimeError(
            "b12x paged indexer cache has an unsupported layout, "
            f"shape={tuple(kv_cache.shape)} stride={tuple(kv_cache.stride())}; "
            f"expected inner strides ({expected_shape_tail[1]}, 1)."
        )

    return kv_cache.as_strided(
        (int(kv_cache.shape[0]), _B12X_COMPRESSED_INDEX_PAGE_WIDTH),
        (int(kv_cache.stride(0)), 1),
    )


def _run_b12x_decode_topk(
    *,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    kv_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    schedule_metadata: torch.Tensor | None,
    topk_indices: torch.Tensor,
    topk_tokens: int,
    active_width: torch.Tensor | None = None,
) -> torch.Tensor:
    """Unified b12x indexer decode top-k for both DSV4 and GLM.

    Both compression modes read the identical page-64 FP8+scale index cache, so
    one b12x-orchestrated entry serves both: b12x sizes the scratch
    (plan_compressed_indexer_scratch), vLLM allocates it (get_simultaneous), and
    index_topk_fp8 owns the internal scorer/top-k routing. ``active_width`` is
    the builder-computed live window (a metadata tensor, not an in-kernel
    reduction); when None, b12x falls back to the capacity cap.
    """
    from b12x.integration.compressed_indexer import (
        COMPRESSED_INDEX_PAGE_SIZE,
        B12XCompressedIndexerScratchCaps,
        index_topk_fp8,
        plan_compressed_indexer_scratch,
    )

    if int(COMPRESSED_INDEX_PAGE_SIZE) != _B12X_COMPRESSED_INDEX_PAGE_SIZE:
        raise RuntimeError(
            "b12x compressed indexer page-size contract changed, got "
            f"{COMPRESSED_INDEX_PAGE_SIZE}; expected "
            f"{_B12X_COMPRESSED_INDEX_PAGE_SIZE}."
        )

    index_k_cache = _flatten_b12x_paged_index_cache(kv_cache)
    expected_num_q_heads = int(q_fp8.shape[1])
    plan = plan_compressed_indexer_scratch(
        B12XCompressedIndexerScratchCaps(
            device=q_fp8.device,
            num_q_heads=expected_num_q_heads,
            max_q_rows=int(q_fp8.shape[0]),
            max_page_table_width=int(block_table.shape[1]),
            topk=int(topk_tokens),
            reserve_paged_logits=False,
        )
    )
    scratch = current_workspace_manager().get_simultaneous(
        *plan.shapes_and_dtypes()
    )
    binding = plan.bind(
        scratch=scratch,
        real_page_table=block_table,
        cache_seqlens_int32=seq_lens,
        active_width=active_width,
        schedule_metadata=schedule_metadata,
        expected_num_q_heads=expected_num_q_heads,
    )
    return index_topk_fp8(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        binding=binding,
        page_size=COMPRESSED_INDEX_PAGE_SIZE,
        expected_num_q_heads=expected_num_q_heads,
        out_indices=topk_indices,
    )


def sparse_attn_indexer(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_quant: torch.Tensor,
    q_scale: torch.Tensor | None,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor,
    skip_k_cache_insert: bool,
    use_fp4_cache: bool = False,
    use_b12x_sparse_indexer: bool = False,
) -> torch.Tensor:
    # careful! this will be None in dummy run
    attn_metadata = get_forward_context().attn_metadata
    fp8_dtype = current_platform.fp8_dtype()
    k_cache_prefix = _resolve_layer_name(k_cache_prefix)

    # assert isinstance(attn_metadata, dict)
    if not isinstance(attn_metadata, dict):
        values_spec, scales_spec = _gather_workspace_shapes(
            total_seq_lens, head_dim, fp8_dtype, use_fp4_cache
        )
        if _b12x_sparse_indexer_requested(use_b12x_sparse_indexer):
            _ensure_b12x_sparse_indexer_supported()
            _reserve_b12x_indexer_extend_worst_case(
                q_quant=q_quant,
                topk_tokens=topk_tokens,
                head_dim=head_dim,
                fp8_dtype=fp8_dtype,
                max_model_len=max_model_len,
            )
        else:
            # Reserve workspace for indexer during profiling run.
            current_workspace_manager().get_simultaneous(
                values_spec, scales_spec, ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8)
            )

            # Dummy allocation to simulate peak logits tensor memory during
            # inference. The B12X path above streams one supertile at a time and
            # has already reserved its fixed scratch via the workspace manager.
            # FP8 elements so elements == bytes.
            max_logits_elems = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024
            _ = torch.empty(
                max_logits_elems, dtype=torch.uint8, device=hidden_states.device
            )

        return sparse_attn_indexer_fake(
            hidden_states,
            k_cache_prefix,
            kv_cache,
            q_quant,
            q_scale,
            k,
            weights,
            quant_block_size,
            scale_fmt,
            topk_tokens,
            head_dim,
            max_model_len,
            total_seq_lens,
            topk_indices_buffer,
            skip_k_cache_insert,
            use_fp4_cache,
            use_b12x_sparse_indexer,
        )
    attn_metadata_narrowed = attn_metadata[k_cache_prefix]
    assert isinstance(attn_metadata_narrowed, DeepseekV32IndexerMetadata)
    slot_mapping = attn_metadata_narrowed.slot_mapping
    has_decode = attn_metadata_narrowed.num_decodes > 0
    has_prefill = attn_metadata_narrowed.num_prefills > 0
    num_decode_tokens = attn_metadata_narrowed.num_decode_tokens

    # q_scale is required iff the FP4 cache path is enabled; the FP8 path
    # folds the Q scale into `weights` inside fused_indexer_q_rope_quant.
    if use_fp4_cache:
        assert q_scale is not None, "use_fp4_cache=True requires q_scale"
    else:
        assert q_scale is None, "q_scale must be None when use_fp4_cache=False"

    # During speculative decoding, k may be padded to the CUDA graph batch
    # size while slot_mapping only covers actual tokens. Truncate k to avoid
    # out-of-bounds reads in the kernel.
    num_tokens = slot_mapping.shape[0]
    if k is not None:
        k = k[:num_tokens]

    if not skip_k_cache_insert:
        # scale_fmt can be None, but the function expects str
        assert scale_fmt is not None
        assert not use_fp4_cache, "Unfused FP4 Insert is not supported yet"
        ops.indexer_k_quant_and_cache(
            k,
            kv_cache,
            slot_mapping,
            quant_block_size,
            scale_fmt,
        )

    topk_indices_buffer[: hidden_states.shape[0]] = -1
    if has_prefill:
        prefill_metadata = attn_metadata_narrowed.prefill
        assert prefill_metadata is not None

        # Layout switches between FP8 (head_dim bytes + 4-byte fp32 scale) and
        # MXFP4 (head_dim/2 bytes packed + head_dim/MXFP4_BLOCK_SIZE ue8m0
        # scales) based on use_fp4_cache.
        use_b12x_indexer = _use_b12x_sparse_indexer(use_b12x_sparse_indexer)
        if use_b12x_indexer and use_fp4_cache:
            raise RuntimeError(
                "b12x sparse indexer currently requires the FP8 indexer cache; "
                "disable use_fp4_indexer_cache or disable b12x sparse indexer."
            )
        b12x_indexer: Any = None
        if use_b12x_indexer:
            b12x_indexer = import_module("b12x.integration.indexer")
        else:
            workspace_manager = current_workspace_manager()
            values_spec, scales_spec = _gather_workspace_shapes(
                total_seq_lens, head_dim, fp8_dtype, use_fp4_cache
            )
            k_quant_full, k_scale_full = workspace_manager.get_simultaneous(
                values_spec,
                scales_spec,
            )
        for chunk in prefill_metadata.chunks:
            q_slice = q_quant[chunk.token_start : chunk.token_end]
            q_scale_slice = (
                q_scale[chunk.token_start : chunk.token_end]
                if q_scale is not None
                else None
            )
            weights_slice = weights[chunk.token_start : chunk.token_end]
            topk_indices = topk_indices_buffer[
                chunk.token_start : chunk.token_end, :topk_tokens
            ]
            if chunk.total_seq_lens <= 0:
                topk_indices.fill_(-1)
                continue

            tile_logits = lengths = topk_values = None
            candidate_values = candidate_indices = None
            topk_indices_out = topk_indices
            if use_b12x_indexer:
                row_has_no_kv = chunk.cu_seqlen_ke <= chunk.cu_seqlen_ks
                b12x_cu_seqlen_ks = torch.where(
                    row_has_no_kv,
                    torch.zeros_like(chunk.cu_seqlen_ks),
                    chunk.cu_seqlen_ks,
                )
                b12x_cu_seqlen_ke = torch.where(
                    row_has_no_kv,
                    torch.ones_like(chunk.cu_seqlen_ke),
                    chunk.cu_seqlen_ke,
                )
                (
                    k_quant,
                    k_scale,
                    tile_logits,
                    lengths,
                    topk_values,
                    topk_indices_out,
                    candidate_values,
                    candidate_indices,
                    merge_positions,
                ) = _get_b12x_indexer_extend_buffers(
                    q_fp8=q_slice,
                    topk_tokens=topk_tokens,
                    total_seq_lens=chunk.total_seq_lens,
                    head_dim=head_dim,
                    fp8_dtype=fp8_dtype,
                    max_k_rows=max_model_len,
                )
            else:
                k_quant = k_quant_full[: chunk.total_seq_lens]
                k_scale = k_scale_full[: chunk.total_seq_lens]

            if not chunk.skip_kv_gather:
                ops.cp_gather_indexer_k_quant_cache(
                    kv_cache,
                    k_quant,
                    k_scale,
                    chunk.block_table,
                    chunk.cu_seq_lens,
                )

            if use_b12x_indexer:
                assert b12x_indexer is not None
                k_scale_f32 = k_scale.view(torch.float32).flatten()
                k_fp8_b12x = (
                    k_quant.view(torch.float8_e4m3fn)
                    if k_quant.dtype == torch.uint8
                    else k_quant
                )
                topk_indices.copy_(
                    _run_b12x_extend_tiled_topk_streaming(
                        q_fp8=q_slice,
                        weights=weights_slice,
                        kv_fp8=(k_fp8_b12x, k_scale_f32),
                        metadata=b12x_indexer.IndexerExtendMetadata(
                            k_start=b12x_cu_seqlen_ks,
                            k_end=b12x_cu_seqlen_ke,
                        ),
                        topk=topk_tokens,
                        contract_phantoms=None,
                        workspace=None,
                        tile_logits=tile_logits,
                        lengths=lengths,
                        output_values=topk_values,
                        output_indices=topk_indices_out,
                        candidate_values=candidate_values,
                        candidate_indices=candidate_indices,
                        merge_positions=merge_positions,
                        supertile_k=_B12X_EXTEND_TOPK_SUPERTILE_K,
                    )
                )
                topk_indices.masked_fill_(row_has_no_kv[:, None], -1)
                _normalize_prefill_topk_to_req_relative(chunk, topk_indices)
                continue

            # DeepGEMM scalar-type tags (zero-copy): MXFP4 values → int8
            # (kPackedFP4), scales → int32 squeezed to 1-D kv_sf / 2-D q_sf.
            if use_fp4_cache:
                q_slice_cast = q_slice.view(torch.int8)
                k_quant_cast = k_quant.view(torch.int8)
                k_scale_cast = k_scale.view(torch.int32).squeeze(-1)
            else:
                q_slice_cast = q_slice
                k_quant_cast = k_quant
                k_scale_cast = k_scale.view(torch.float32).squeeze(-1)
            if current_platform.is_xpu():
                if q_scale_slice is not None:
                    raise RuntimeError("XPU fp8_mqa_logits does not support FP4 Q")
                logits = torch.ops.vllm.xpu_fp8_mqa_logits(
                    q_slice_cast,
                    k_quant_cast,
                    k_scale_cast,
                    weights_slice,
                    chunk.cu_seqlen_ks,
                    chunk.cu_seqlen_ke,
                )
            else:
                from vllm.utils.deep_gemm import fp8_fp4_mqa_logits

                logits = fp8_fp4_mqa_logits(
                    (q_slice_cast, q_scale_slice),
                    (k_quant_cast, k_scale_cast),
                    weights_slice,
                    chunk.cu_seqlen_ks,
                    chunk.cu_seqlen_ke,
                    clean_logits=False,
                )
            num_rows = logits.shape[0]

            ops.top_k_per_row_prefill(
                logits,
                chunk.cu_seqlen_ks,
                chunk.cu_seqlen_ke,
                topk_indices,
                num_rows,
                logits.stride(0),
                logits.stride(1),
                topk_tokens,
            )

    if has_decode:
        decode_metadata = attn_metadata_narrowed.decode
        assert decode_metadata is not None
        use_b12x_indexer = _use_b12x_sparse_indexer(use_b12x_sparse_indexer)
        if use_b12x_indexer and use_fp4_cache:
            raise RuntimeError(
                "b12x sparse indexer currently requires the FP8 indexer cache; "
                "disable use_fp4_indexer_cache or disable b12x sparse indexer."
            )

        b12x_seq_lens = decode_metadata.seq_lens
        b12x_block_table = decode_metadata.block_table
        if b12x_seq_lens.dim() == 2:
            b12x_batch_size, b12x_next_n = b12x_seq_lens.shape
            if num_decode_tokens == b12x_batch_size * b12x_next_n:
                b12x_seq_lens = b12x_seq_lens.reshape(-1).contiguous()
                b12x_block_table = b12x_block_table.repeat_interleave(
                    b12x_next_n, dim=0
                ).contiguous()
        b12x_decode_supported = (
            use_b12x_indexer
            and not decode_metadata.requires_padding
            and b12x_seq_lens.dim() == 1
        )
        if use_b12x_indexer and (
            decode_metadata.requires_padding or b12x_seq_lens.dim() != 1
        ):
            raise RuntimeError(
                "b12x sparse indexer decode requires an unpadded rank-1 "
                "seq_lens contract after native-spec normalization; refusing "
                "to fall back to DeepGEMM. "
                f"requires_padding={decode_metadata.requires_padding}, "
                f"seq_lens_shape={tuple(decode_metadata.seq_lens.shape)}, "
                f"normalized_seq_lens_shape={tuple(b12x_seq_lens.shape)}, "
                f"num_decode_tokens={num_decode_tokens}."
            )

        if b12x_decode_supported:
            # Prefix slice of an already-contiguous buffer stays contiguous
            # (b12x_seq_lens/b12x_block_table are normalized contiguous upstream),
            # so .contiguous() here was a guaranteed no-op per decoded token.
            seq_lens = b12x_seq_lens[:num_decode_tokens]
            block_table = b12x_block_table[:num_decode_tokens]
            topk_indices = topk_indices_buffer[:num_decode_tokens, :topk_tokens]
            # One unified b12x decode path for both DSV4 (compress_ratio>1) and
            # GLM (compress_ratio==1): the kernel byte layout is identical, so
            # compress_ratio only shapes the seq_lens/active_width the builder
            # already prepared. b12x owns the scorer/top-k routing internally.
            _run_b12x_decode_topk(
                q_fp8=q_quant[:num_decode_tokens].contiguous(),
                weights=weights[:num_decode_tokens].contiguous(),
                kv_cache=kv_cache,
                seq_lens=seq_lens,
                block_table=block_table,
                schedule_metadata=decode_metadata.schedule_metadata,
                active_width=decode_metadata.active_width,
                topk_indices=topk_indices,
                topk_tokens=topk_tokens,
            )
            return topk_indices_buffer

        schedule_metadata = decode_metadata.schedule_metadata
        if schedule_metadata is None:
            raise RuntimeError(
                "DeepGEMM/XPU sparse indexer decode requires schedule metadata; "
                "enable VLLM_USE_B12X_SPARSE_INDEXER for the b12x path or check "
                "the indexer metadata builder."
            )

        kv_cache = kv_cache_as_quant_view(kv_cache, head_dim, use_fp4_cache)
        decode_lens = decode_metadata.decode_lens
        if decode_metadata.requires_padding:
            # pad in edge case where we have short chunked prefill length <
            # decode_threshold since we unstrictly split
            # prefill and decode by decode_threshold
            # (currently set to 1 + speculative tokens).
            # FP8 Q is float8_e4m3fn (pack_seq_triton's fp32 pad path is OK —
            # downstream context_lens masks stale slots). MXFP4 Q is two
            # uint8 tensors (values + ue8m0 scales) — use the dedicated uint8
            # packer with pad_byte=0 so padded slots dequantize to 0 and
            # can't produce NaN/Inf in the logits kernel.
            if q_scale is not None:
                padded_q_quant_decode_tokens = pack_seq_triton(
                    q_quant[:num_decode_tokens], decode_lens, pad_value=0
                )
                padded_q_scale = pack_seq_triton(
                    q_scale[:num_decode_tokens], decode_lens, pad_value=0
                )
            else:
                padded_q_quant_decode_tokens = pack_seq_triton(
                    q_quant[:num_decode_tokens], decode_lens
                )
                padded_q_scale = None
        else:
            padded_q_quant_decode_tokens = q_quant[:num_decode_tokens].reshape(
                decode_lens.shape[0], -1, *q_quant.shape[1:]
            )
            if q_scale is not None:
                padded_q_scale = q_scale[:num_decode_tokens].reshape(
                    decode_lens.shape[0], -1, *q_scale.shape[1:]
                )
            else:
                padded_q_scale = None
        # TODO: move and optimize below logic with triton kernels
        batch_size = padded_q_quant_decode_tokens.shape[0]
        next_n = padded_q_quant_decode_tokens.shape[1]
        num_padded_tokens = batch_size * next_n
        seq_lens = decode_metadata.seq_lens[:batch_size]
        # seq_lens is always 2D: (B, next_n) for native spec decode, (B, 1)
        # otherwise. deep_gemm fp8_fp4_paged_mqa_logits requires 2D context_lens;
        # the downstream topk kernels accept both 1D and 2D.
        padded_q_quant_cast = (
            padded_q_quant_decode_tokens.view(torch.int8)
            if use_fp4_cache
            else padded_q_quant_decode_tokens
        )
        if current_platform.is_xpu():
            if padded_q_scale is not None:
                raise RuntimeError("XPU fp8_paged_mqa_logits does not support FP4 Q")
            seq_lens_xpu = (
                seq_lens[:, -1].contiguous() if seq_lens.ndim == 2 else seq_lens
            )
            logits = torch.ops.vllm.xpu_fp8_paged_mqa_logits(
                padded_q_quant_cast,
                kv_cache,
                weights[:num_padded_tokens],
                seq_lens_xpu,
                decode_metadata.block_table,
                schedule_metadata,
                max_model_len,
            )
        else:
            from vllm.utils.deep_gemm import fp8_fp4_paged_mqa_logits

            logits = fp8_fp4_paged_mqa_logits(
                (padded_q_quant_cast, padded_q_scale),
                kv_cache,
                weights[:num_padded_tokens],
                seq_lens,
                decode_metadata.block_table,
                schedule_metadata,
                max_model_len=max_model_len,
                clean_logits=False,
            )
        num_rows = logits.shape[0]
        topk_indices = topk_indices_buffer[:num_padded_tokens, :topk_tokens]

        if current_platform.is_cuda() and topk_tokens in (512, 1024, 2048):
            workspace_manager = current_workspace_manager()
            (topk_workspace,) = workspace_manager.get_simultaneous(
                ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
            )
            torch.ops._C.persistent_topk(
                logits,
                seq_lens,
                topk_indices,
                topk_workspace,
                topk_tokens,
                attn_metadata_narrowed.max_seq_len,
            )
        else:
            ops.top_k_per_row_decode(
                logits,
                next_n,
                seq_lens,
                topk_indices,
                num_rows,
                logits.stride(0),
                logits.stride(1),
                topk_tokens,
            )

        if decode_metadata.requires_padding:
            # if padded, we need to unpack
            # the topk indices removing padded tokens
            topk_indices = unpack_seq_triton(
                topk_indices.reshape(batch_size, -1, topk_indices.shape[-1]),
                decode_lens,
            )
            topk_indices_buffer[: topk_indices.shape[0], : topk_indices.shape[-1]] = (
                topk_indices
            )

    return topk_indices_buffer


def sparse_attn_indexer_fake(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_quant: torch.Tensor,
    q_scale: torch.Tensor | None,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor | None,
    skip_k_cache_insert: bool,
    use_fp4_cache: bool = False,
    use_b12x_sparse_indexer: bool = False,
) -> torch.Tensor:
    return topk_indices_buffer


direct_register_custom_op(
    op_name="sparse_attn_indexer",
    op_func=sparse_attn_indexer,
    mutates_args=["topk_indices_buffer"],
    fake_impl=sparse_attn_indexer_fake,
    dispatch_key=current_platform.dispatch_key,
)


@CustomOp.register("sparse_attn_indexer")
class SparseAttnIndexer(CustomOp):
    """Sparse Attention Indexer Custom Op Layer. This layer is extracted as a
    separate custom op since it involves heavy custom kernels like `mqa_logits`,
    `paged_mqa_logits` and `top_k_per_row`, etc. Those kernels maybe requires
    specific memory layout or implementation for different hardware backends to
    achieve optimal performance.

    For now, the default native path will use CUDA backend path. Other platform
    may requires add the corresponding Custom Op name `sparse_attn_indexer` to
    `custom_ops` in `CompilationConfig` to enable the platform specific path.
    """

    def __init__(
        self,
        k_cache,
        quant_block_size: int,
        scale_fmt: str,
        topk_tokens: int,
        head_dim: int,
        max_model_len: int,
        max_total_seq_len: int,
        topk_indices_buffer: torch.Tensor,
        skip_k_cache_insert: bool = False,
        use_fp4_cache: bool = False,
    ):
        super().__init__()
        self.k_cache = k_cache
        self.quant_block_size = quant_block_size
        self.scale_fmt = scale_fmt
        self.topk_tokens = topk_tokens
        self.head_dim = head_dim
        self.max_model_len = max_model_len
        self.max_total_seq_len = max_total_seq_len
        self.topk_indices_buffer = topk_indices_buffer
        self.skip_k_cache_insert = skip_k_cache_insert
        self.use_fp4_cache = use_fp4_cache
        self.use_b12x_sparse_indexer = use_b12x_sparse_indexer()
        if self.use_b12x_sparse_indexer:
            if self.use_fp4_cache:
                raise RuntimeError(
                    "B12X sparse indexer/top-k requires the FP8/C4 indexer "
                    "cache; disable use_fp4_indexer_cache."
                )
        elif current_platform.is_cuda() and not has_deep_gemm():
            raise RuntimeError(
                "Sparse Attention Indexer CUDA op requires DeepGEMM to be installed."
            )

    def forward_native(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        if current_platform.is_cuda() or current_platform.is_xpu():
            return self.forward_cuda(hidden_states, q_quant, k, weights)
        elif current_platform.is_rocm():
            return self.forward_hip(hidden_states, q_quant, k, weights)
        else:
            raise NotImplementedError(
                "SparseAttnIndexer native forward is only implemented for "
                "CUDA, ROCm and XPU platforms."
            )

    def forward_cuda(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        # FP8 path: single tensor (per-token scale is folded into `weights`).
        # FP4 path: (values, scales) tuple with scales required by the kernel.
        if isinstance(q_quant, tuple):
            q_values, q_scale = q_quant
        else:
            q_values, q_scale = q_quant, None
        return torch.ops.vllm.sparse_attn_indexer(
            hidden_states,
            _encode_layer_name(self.k_cache.prefix),
            self.k_cache.kv_cache,
            q_values,
            q_scale,
            k,
            weights,
            self.quant_block_size,
            self.scale_fmt,
            self.topk_tokens,
            self.head_dim,
            self.max_model_len,
            self.max_total_seq_len,
            self.topk_indices_buffer,
            self.skip_k_cache_insert,
            self.use_fp4_cache,
            self.use_b12x_sparse_indexer,
        )

    def forward_xpu(
        self,
        hidden_states: torch.Tensor,
        q_fp8: torch.Tensor,
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        return self.forward_cuda(hidden_states, q_fp8, k, weights)

    def forward_hip(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        assert not self.use_fp4_cache, "AMD platform doesn't support fp4 cache yet"
        assert isinstance(q_quant, torch.Tensor), (
            "AMD sparse_attn_indexer expects a single FP8 q_quant tensor"
        )
        if rocm_aiter_ops.is_enabled():
            return torch.ops.vllm.rocm_aiter_sparse_attn_indexer(
                hidden_states,
                _encode_layer_name(self.k_cache.prefix),
                self.k_cache.kv_cache,
                q_quant,
                k,
                weights,
                self.quant_block_size,
                self.scale_fmt,
                self.topk_tokens,
                self.head_dim,
                self.max_model_len,
                self.max_total_seq_len,
                self.topk_indices_buffer,
                skip_k_cache_insert=self.skip_k_cache_insert,
            )
        raise RuntimeError(
            "Sparse attention indexer ROCm path is only supported on AITER. "
            "Please enable aiter with VLLM_ROCM_USE_AITER=1"
        )
