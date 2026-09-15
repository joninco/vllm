# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU references for native FP8 prefill chunks from the production fused writer.

Inputs are BF16 activations, real fused RMS/RoPE outputs, paged native caches,
and request/interleave metadata. Native-copy output must equal the producer's
stored bytes, including FP32 scales; a BF16 re-quantization is a distinct
computation. These single-device references emulate DCP ownership without
validating distributed collectives. Only the GPU coordinator runs them.
"""

import pytest
import torch

from vllm.models.deepseek_v32.common.kernels import fused_norm_rope

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires coordinator-owned CUDA GPU"
)


RECORD_BYTES = 656
PAGE_SIZE = 64
DCP_SIZE = 4


def _produce_native(latent, rope, positions, slots, cache):
    """Invoke the serving fused producer; its bytes are the numerical oracle."""
    device = latent.device
    rows = latent.shape[0]
    angles = (
        torch.arange(256, device=device, dtype=torch.float32)[:, None]
        * (torch.arange(32, device=device, dtype=torch.float32)[None, :] + 1)
        / 1024
    )
    cos_sin = torch.cat((angles.cos(), angles.sin()), dim=1)
    normalized = torch.empty_like(latent)
    rotated = torch.empty_like(rope)
    fused_norm_rope(
        positions=positions,
        q_c=torch.ones((rows, 512), dtype=torch.bfloat16, device=device),
        q_rms_norm_w=torch.ones(512, dtype=torch.bfloat16, device=device),
        q_rms_eps=1e-5,
        kv_c=latent,
        kv_rms_norm_w=torch.ones(512, dtype=torch.bfloat16, device=device),
        kv_rms_eps=1e-5,
        k_pe=rope,
        k_rope_cos_sin_cache=cos_sin,
        index_k=None,
        index_k_layer_norm_w=None,
        index_k_layer_norm_bias=None,
        index_k_layer_norm_eps=1e-5,
        index_k_rope_cos_sin_cache=None,
        topk_indices_buffer=torch.empty((rows, 2048), dtype=torch.int32, device=device),
        slot_mapping=slots,
        mla_kv_cache=cache,
        mla_kv_cache_dtype="fp8_ds_mla",
        has_indexer=False,
        kv_c_out=normalized,
        k_pe_out=rotated,
        materialize_nonlocal_mla_inputs=True,
    )
    return normalized, rotated


def test_bf16_materialization_cannot_reconstruct_fused_fp32_native_records():
    from b12x.attention._shared.mla.kv_cache import concat_and_cache_fp8_ds_mla

    device = torch.device("cuda")
    slots = torch.arange(4, dtype=torch.int64, device=device)
    positions = torch.arange(4, dtype=torch.int64, device=device)
    native = torch.zeros((1, PAGE_SIZE, RECORD_BYTES), dtype=torch.uint8, device=device)
    requantized = torch.zeros_like(native)
    # Constant rows isolate the FP32 normalization residual rounded away by BF16.
    # Small nonuniform perturbations additionally exercise FP8 boundary changes.
    for change in (0, 1):
        latent = torch.ones((4, 512), dtype=torch.bfloat16, device=device)
        latent[1, ::7] = 1.0078125 + change * 0.0078125
        latent[2, ::3] = -0.99609375 + change * 0.00390625
        latent[3, ::5] = 2.015625 + change * 0.015625
        rope = torch.full(
            (4, 64), 0.5 + change * 0.25, dtype=torch.bfloat16, device=device
        )
        normalized, rotated = _produce_native(latent, rope, positions, slots, native)
        concat_and_cache_fp8_ds_mla(normalized, rotated, requantized, slots)
        # The actual producer quantizes FP32 registers, not this rounded tensor.
        assert not torch.equal(native[0, :4, :528], requantized[0, :4, :528])
        assert not torch.equal(native[0, 0, 512:528], requantized[0, 0, 512:528])
        torch.testing.assert_close(
            native[0, :4, 528:], requantized[0, :4, 528:], rtol=0, atol=0
        )


def _local_count(length, rank, interleave):
    cycles, remainder = divmod(length, DCP_SIZE * interleave)
    return cycles * interleave + min(interleave, max(0, remainder - rank * interleave))


@pytest.mark.parametrize("interleave", [1, 4, 64])
def test_current_chunk_copy_preserves_real_producer_bytes_across_owners(interleave):
    from b12x.attention._shared.mla.kv_cache import (
        gather_ckv_current_chunk,
        insert_ckv_current_chunk,
    )

    device = torch.device("cuda")
    lengths = [73, 150]
    query_lengths = [17, 65]
    starts = [0, 17, 82]
    rows = starts[-1]
    global_lens = torch.tensor(lengths, dtype=torch.int32, device=device)
    query_starts = torch.tensor(starts, dtype=torch.int32, device=device)
    positions_cpu = [
        p
        for length, query in zip(lengths, query_lengths)
        for p in range(length - query, length)
    ]
    positions = torch.tensor(positions_cpu, dtype=torch.int64, device=device)
    request_ids = [0] * query_lengths[0] + [1] * query_lengths[1]
    rank_starts_cpu = []
    for rank in range(DCP_SIZE):
        first = _local_count(lengths[0], rank, interleave)
        rank_starts_cpu.append([0, first])
    rank_starts = torch.tensor(rank_starts_cpu, dtype=torch.int32, device=device)
    # A live padded rank span differs deliberately from planned return capacity.
    padded_tokens = 192
    planned_rank_capacity = 256
    current_capacity = 96
    gathered_current = torch.empty(
        (DCP_SIZE * current_capacity, RECORD_BYTES), dtype=torch.uint8, device=device
    )
    full = torch.empty(
        (DCP_SIZE * planned_rank_capacity // PAGE_SIZE, PAGE_SIZE, RECORD_BYTES),
        dtype=torch.uint8,
        device=device,
    )
    oracle_cache = torch.empty(
        (2, PAGE_SIZE, RECORD_BYTES), dtype=torch.uint8, device=device
    )
    oracle_slots = torch.arange(rows, dtype=torch.int64, device=device)
    table_cpu = [[3, 1, 5], [0, 4, 2]]
    table = torch.tensor(table_cpu, dtype=torch.int32, device=device)
    source = torch.empty((6, PAGE_SIZE, RECORD_BYTES), dtype=torch.uint8, device=device)
    pointer = full.data_ptr()
    generator = torch.Generator().manual_seed(67091)
    for change in (0, 1):
        latent = torch.randn((rows, 512), generator=generator, dtype=torch.float32).to(
            device=device, dtype=torch.bfloat16
        )
        rope = torch.randn((rows, 64), generator=generator, dtype=torch.float32).to(
            device=device, dtype=torch.bfloat16
        )
        oracle_cache.fill_(19)
        _produce_native(latent, rope, positions, oracle_slots, oracle_cache)
        gathered_current.fill_(231)
        full.fill_(91 + change)
        expected = full.clone().view(-1, RECORD_BYTES)
        for rank in range(DCP_SIZE):
            owned = [
                row
                for row, pos in enumerate(positions_cpu)
                if (pos // interleave) % DCP_SIZE == rank
            ]
            physical_slots = []
            for row in owned:
                pos, req = positions_cpu[row], request_ids[row]
                local = pos // (DCP_SIZE * interleave) * interleave + pos % interleave
                physical_slots.append(
                    table_cpu[req][local // PAGE_SIZE] * PAGE_SIZE + local % PAGE_SIZE
                )
                expected[
                    rank * padded_tokens + rank_starts_cpu[rank][req] + local
                ].copy_(oracle_cache.view(-1, RECORD_BYTES)[row])
            source.fill_(203)
            if owned:
                row_ids = torch.tensor(owned, dtype=torch.int64, device=device)
                _produce_native(
                    latent[row_ids],
                    rope[row_ids],
                    positions[row_ids],
                    torch.tensor(physical_slots, dtype=torch.int64, device=device),
                    source,
                )
            gather_ckv_current_chunk(
                source,
                gathered_current[
                    rank * current_capacity : (rank + 1) * current_capacity
                ],
                table,
                global_lens,
                query_starts,
                dcp_rank=rank,
                dcp_world_size=DCP_SIZE,
                interleave=interleave,
                num_reqs=len(lengths),
                current_capacity=current_capacity,
            )
        insert_ckv_current_chunk(
            gathered_current,
            full,
            rank_starts,
            global_lens,
            query_starts,
            dcp_world_size=DCP_SIZE,
            interleave=interleave,
            num_reqs=len(lengths),
            current_capacity=current_capacity,
            padded_tokens=padded_tokens,
        )
        torch.testing.assert_close(
            full.view(-1, RECORD_BYTES), expected, rtol=0, atol=0
        )
        assert full.data_ptr() == pointer
