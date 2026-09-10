# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU references for the step-local index K copy written by the fused producer.

Under DCP each rank stores index keys only for the tokens it owns. A prefill
request whose whole context is among the step's tokens is additionally
written, token by token and independent of ownership, into a step-local copy
with the persistent cache's page layout. These references launch the serving
fused norm/RoPE kernel and compare the copy's bytes, including FP32 scales,
with the bytes the same kernel writes for an owning rank. Only the GPU
coordinator runs them.
"""

import pytest
import torch

from vllm.models.deepseek_v32.common.kernels import fused_norm_rope

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires coordinator-owned CUDA GPU"
)

PAGE_SIZE = 64
INDEX_DIM = 128
RECORD_BYTES = INDEX_DIM + 4
ROTARY_DIM = 64
DCP_SIZE = 4


def _cos_sin(device):
    angles = (
        torch.arange(512, device=device, dtype=torch.float32)[:, None]
        * (
            torch.arange(ROTARY_DIM // 2, device=device, dtype=torch.float32)[None, :]
            + 1
        )
        / 1024
    )
    return torch.cat((angles.cos(), angles.sin()), dim=1)


def _produce(rows, shard_slots, local_slots, local_cache, *, has_indexer=True):
    """Run the fused producer over ``rows`` tokens and return the shard cache."""
    device = shard_slots.device
    generator = torch.Generator(device=device).manual_seed(7)
    index_k = torch.randn(
        (rows, INDEX_DIM), device=device, dtype=torch.bfloat16, generator=generator
    )
    pages = -(-rows // PAGE_SIZE)
    shard_cache = torch.zeros(
        (pages, PAGE_SIZE, RECORD_BYTES), dtype=torch.uint8, device=device
    )
    fused_norm_rope(
        positions=torch.arange(rows, device=device, dtype=torch.int64),
        q_c=torch.ones((rows, 512), dtype=torch.bfloat16, device=device),
        q_rms_norm_w=torch.ones(512, dtype=torch.bfloat16, device=device),
        q_rms_eps=1e-5,
        kv_c=torch.ones((rows, 512), dtype=torch.bfloat16, device=device),
        kv_rms_norm_w=torch.ones(512, dtype=torch.bfloat16, device=device),
        kv_rms_eps=1e-5,
        k_pe=torch.ones((rows, ROTARY_DIM), dtype=torch.bfloat16, device=device),
        k_rope_cos_sin_cache=_cos_sin(device),
        index_k=index_k,
        index_k_layer_norm_w=torch.linspace(0.5, 1.5, INDEX_DIM, device=device).to(
            torch.bfloat16
        ),
        index_k_layer_norm_bias=torch.linspace(-0.1, 0.1, INDEX_DIM, device=device).to(
            torch.bfloat16
        ),
        index_k_layer_norm_eps=1e-6,
        index_k_rope_cos_sin_cache=_cos_sin(device),
        topk_indices_buffer=torch.empty((rows, 2048), dtype=torch.int32, device=device),
        slot_mapping=shard_slots,
        indexer_k_cache=shard_cache,
        has_indexer=has_indexer,
        indexer_local_cache=local_cache,
        indexer_local_slot_mapping=local_slots,
    )
    torch.accelerator.synchronize()
    return shard_cache


def _records(cache, slots):
    """Per-token bytes of ``cache`` for the nonnegative ``slots``, in slot order.

    A page stores its 64 FP8 key rows first and their 64 FP32 UE8M0 scales
    after them; a token's record is its 128 key bytes followed by its 4 scale
    bytes.
    """
    slots = slots[slots >= 0].long()
    pages = cache.view(cache.shape[0], -1)
    page, offset = slots // PAGE_SIZE, slots % PAGE_SIZE
    keys = torch.arange(INDEX_DIM, device=cache.device)
    key_bytes = pages[page[:, None], offset[:, None] * INDEX_DIM + keys[None, :]]
    scale_bytes = pages[
        page[:, None],
        PAGE_SIZE * INDEX_DIM
        + offset[:, None] * 4
        + torch.arange(4, device=cache.device)[None, :],
    ]
    return torch.cat((key_bytes, scale_bytes), dim=1)


@pytest.mark.parametrize("rows", [1, 64, 200])
def test_local_copy_equals_owning_rank_bytes_for_every_token(rows):
    device = torch.device("cuda")
    token = torch.arange(rows, device=device, dtype=torch.int32)
    # Rank 1 of four with interleave 1 owns every fourth token.
    owned = token % DCP_SIZE == 1
    shard_slots = torch.where(owned, token, torch.full_like(token, -1))
    local_slots = token.clone()
    pages = -(-rows // PAGE_SIZE)
    local_cache = torch.zeros(
        (pages, PAGE_SIZE, RECORD_BYTES), dtype=torch.uint8, device=device
    )
    shard_cache = _produce(rows, shard_slots, local_slots, local_cache)

    # The same launch on a rank owning every token is the byte oracle.
    reference = _records(_produce(rows, token.clone(), None, None), token)
    assert reference.any(dim=1).all()
    assert torch.equal(_records(local_cache, local_slots), reference)
    # Ownership still bounds the persistent shard write.
    assert torch.equal(_records(shard_cache, shard_slots), reference[owned])
    assert not _records(shard_cache, token)[~owned].any()


def test_tokens_without_a_local_slot_keep_the_sharded_write_only():
    device = torch.device("cuda")
    rows = 96
    token = torch.arange(rows, device=device, dtype=torch.int32)
    owned = token % DCP_SIZE == 0
    shard_slots = torch.where(owned, token, torch.full_like(token, -1))
    # The second half of the step carries history and gets no local slot.
    fresh = token < 48
    local_slots = torch.where(fresh, token, torch.full_like(token, -1))
    local_cache = torch.zeros(
        (2, PAGE_SIZE, RECORD_BYTES), dtype=torch.uint8, device=device
    )
    shard_cache = _produce(rows, shard_slots, local_slots, local_cache)

    reference = _records(_produce(rows, token.clone(), None, None), token)
    copy = _records(local_cache, token)
    assert torch.equal(copy[fresh], reference[fresh])
    assert not copy[~fresh].any()
    assert torch.equal(_records(shard_cache, shard_slots), reference[owned])


def test_shared_layers_ignore_the_local_copy():
    device = torch.device("cuda")
    rows = 32
    token = torch.arange(rows, device=device, dtype=torch.int32)
    local_cache = torch.zeros(
        (1, PAGE_SIZE, RECORD_BYTES), dtype=torch.uint8, device=device
    )
    shard_cache = _produce(
        rows, token.clone(), token.clone(), local_cache, has_indexer=False
    )
    assert not local_cache.any()
    assert not shard_cache.any()
