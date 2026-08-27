# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.models.glm5next.nvidia import pooled_indexer as pooled_indexer_module
from vllm.models.glm5next.nvidia.pooled_indexer import Glm5NextPooledIndexer


def _padded_main_cache(
    *, pages: int = 3, block_size: int = 64
) -> tuple[torch.Tensor, torch.Tensor, int]:
    record_bytes = 528
    tail_bytes = block_size // 4 * 128 * torch.bfloat16.itemsize
    page_bytes = block_size * record_bytes + tail_bytes
    storage = torch.zeros(pages * page_bytes, dtype=torch.uint8)
    main_cache = torch.as_strided(
        storage,
        size=(pages, block_size, record_bytes),
        stride=(page_bytes, record_bytes, 1),
    )
    return storage, main_cache, page_bytes


def test_glm5next_selector_lazily_caches_fp32_head_projection() -> None:
    hidden_size = 8
    indexer = Glm5NextPooledIndexer.__new__(Glm5NextPooledIndexer)
    nn.Module.__init__(indexer)
    indexer.weights_proj = nn.Linear(hidden_size, 32, bias=False, dtype=torch.bfloat16)
    with torch.no_grad():
        values = torch.arange(32 * hidden_size, dtype=torch.float32).view(
            32, hidden_size
        )
        indexer.weights_proj.weight.copy_(values.to(torch.bfloat16) / 128)
    hidden = (
        torch.linspace(-1, 1, 3 * hidden_size, dtype=torch.float32)
        .view(3, hidden_size)
        .to(torch.bfloat16)
    )

    assert not hasattr(indexer, "_weights_proj_fp32")
    expected = torch.nn.functional.linear(
        hidden.float(), indexer.weights_proj.weight.float()
    )
    actual = indexer._project_head_weights(hidden)
    torch.testing.assert_close(actual, expected)
    assert indexer._weights_proj_fp32.dtype == torch.float32
    cache_ptr = indexer._weights_proj_fp32.data_ptr()

    torch.testing.assert_close(indexer._project_head_weights(hidden), expected)
    assert indexer._weights_proj_fp32.data_ptr() == cache_ptr


def test_glm5next_selector_restores_speculative_interval_starts() -> None:
    indexer = Glm5NextPooledIndexer.__new__(Glm5NextPooledIndexer)
    nn.Module.__init__(indexer)
    indexer.register_buffer(
        "_raw_interval_start_positions", torch.tensor([31, 47, 63]), persistent=False
    )
    indexer.register_buffer(
        "_raw_interval_start_snapshot",
        torch.empty(3, dtype=torch.int64),
        persistent=False,
    )

    indexer.snapshot_speculative_interval_starts()
    indexer._raw_interval_start_positions.copy_(torch.tensor([36, 52, 68]))
    indexer.restore_speculative_interval_starts()

    assert torch.equal(
        indexer._raw_interval_start_positions, torch.tensor([31, 47, 63])
    )


def test_glm5next_compressed_cache_view_aliases_padded_page_tail() -> None:
    storage, main_cache, page_bytes = _padded_main_cache()

    compressed = Glm5NextPooledIndexer._compressed_cache_view(main_cache)

    assert compressed.shape == (3, 16, 128)
    assert compressed.stride() == (page_bytes // 2, 128, 1)
    compressed[0].fill_(1)
    compressed[1].fill_(2)
    storage_bf16 = storage.view(torch.bfloat16)
    semantic_elements = 64 * 528 // 2
    page_elements = page_bytes // 2
    assert torch.all(storage_bf16[:semantic_elements] == 0)
    assert torch.all(
        storage_bf16[semantic_elements : semantic_elements + 16 * 128] == 1
    )
    assert torch.all(
        storage_bf16[
            page_elements + semantic_elements : page_elements
            + semantic_elements
            + 16 * 128
        ]
        == 2
    )


def test_glm5next_compressed_cache_view_requires_selector_tail() -> None:
    main_cache = torch.empty((2, 64, 528), dtype=torch.uint8)

    with pytest.raises(ValueError, match="does not contain the selector tail"):
        Glm5NextPooledIndexer._compressed_cache_view(main_cache)


def test_glm5next_compressed_cache_view_handles_interleaved_layers() -> None:
    pages = 3
    layers = 11
    layer = 5
    block_size = 64
    record_bytes = 528
    tail_elements = block_size // 4 * 128
    page_bytes = block_size * record_bytes + tail_elements * 2
    physical_page_bytes = layers * page_bytes
    storage = torch.zeros(pages * physical_page_bytes, dtype=torch.uint8)
    main_cache = torch.as_strided(
        storage,
        size=(pages, block_size, record_bytes),
        stride=(physical_page_bytes, record_bytes, 1),
        storage_offset=layer * page_bytes,
    )

    compressed = Glm5NextPooledIndexer._compressed_cache_view(main_cache)

    assert compressed.shape == (pages, block_size // 4, 128)
    assert compressed.stride() == (physical_page_bytes // 2, 128, 1)
    compressed[2].fill_(7)
    tail_start_bytes = (
        2 * physical_page_bytes + layer * page_bytes + block_size * record_bytes
    )
    tail = storage[tail_start_bytes : tail_start_bytes + tail_elements * 2]
    assert torch.all(tail.view(torch.bfloat16) == 7)


def test_glm5next_bind_uses_aligned_manager_page_geometry(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakePlan:
        def scratch_specs(self):
            return (SimpleNamespace(shape=(64,), dtype=torch.uint8),)

        def bind(self, **kwargs):
            captured["binding"] = kwargs
            return SimpleNamespace()

    class FakeModule:
        @staticmethod
        def is_supported(device):
            return device.type == "cpu"

        @staticmethod
        def Caps(**kwargs):
            captured["caps"] = kwargs
            return SimpleNamespace(**kwargs)

        @staticmethod
        def plan(caps):
            captured["planned_caps"] = caps
            return FakePlan()

    monkeypatch.setattr(
        pooled_indexer_module,
        "get_b12x_glm_pooled_indexer",
        lambda: FakeModule,
    )
    indexer = Glm5NextPooledIndexer.__new__(Glm5NextPooledIndexer)
    nn.Module.__init__(indexer)
    indexer.max_seqs = 2
    indexer.max_tokens = 4
    indexer.max_model_len = 1_048_576
    indexer.max_speculative_tokens = 5
    indexer.block_size = 64
    indexer._compressed_table_width = math.ceil(indexer.max_model_len / 64)
    indexer._compressed_block_table = torch.full(
        (2, indexer._compressed_table_width), -1, dtype=torch.int32
    )
    indexer._raw_k_ring = torch.empty((2, 12, 128), dtype=torch.bfloat16)
    indexer._raw_gate_ring = torch.empty((2, 12, 128), dtype=torch.bfloat16)
    indexer._raw_logical_positions = torch.full((2, 12), -1, dtype=torch.int64)
    indexer._raw_interval_start_positions = torch.full((2,), -1, dtype=torch.int64)
    indexer._raw_state_slot_ids = torch.full((2,), -1, dtype=torch.int32)
    indexer._sequence_lengths = torch.zeros(2, dtype=torch.int32)
    indexer._decode_query_start_loc = torch.zeros(3, dtype=torch.int32)
    indexer._prefill_query_start_loc = torch.zeros(3, dtype=torch.int32)
    indexer._prefill_request_ids = torch.empty(4, dtype=torch.int32)
    indexer._num_accepted_tokens = torch.ones(2, dtype=torch.int32)
    indexer._reset_mask = torch.zeros(2, dtype=torch.bool)
    indexer._prefix_lengths = torch.zeros(2, dtype=torch.int32)
    indexer.index_kpool_compress_ape = nn.Parameter(
        torch.empty((4, 128), dtype=torch.bfloat16)
    )
    indexer.topk_indices_buffer = torch.empty((4, 2051), dtype=torch.int32)

    _, main_cache, _ = _padded_main_cache(pages=3, block_size=2304)
    indexer.bind_main_kv_cache(main_cache)

    expected_width = math.ceil(indexer.max_model_len / 2304)
    caps = captured["caps"]
    assert isinstance(caps, dict)
    assert caps["compressed_page_size"] == 576
    assert caps["num_compressed_cache_pages"] == expected_width
    assert indexer.block_size == 2304
    assert indexer._compressed_table_width == expected_width
    assert indexer._compressed_block_table.shape == (2, expected_width)
    binding = captured["binding"]
    assert isinstance(binding, dict)
    assert binding["compressed_k_cache"].shape == (3, 576, 128)
    assert binding["compressed_block_table"].is_contiguous()

    block_table = torch.arange(2 * expected_width, dtype=torch.int32).view(
        2, expected_width
    )
    metadata = SimpleNamespace(
        num_reqs=2,
        block_table=block_table,
        seq_lens=torch.tensor([1, 2], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        num_decodes=2,
        num_decode_tokens=2,
        selector_state_slot_ids=torch.tensor([0, 1], dtype=torch.int32),
        selector_state_is_fresh=torch.tensor([True, False]),
        selector_num_accepted_tokens=torch.tensor([1, 2], dtype=torch.int32),
    )
    indexer._stage_metadata(metadata, rows=2)
    assert torch.equal(indexer._compressed_block_table, block_table)


def test_glm5next_mixed_prefill_metadata_uses_call_local_request_rows() -> None:
    indexer = Glm5NextPooledIndexer.__new__(Glm5NextPooledIndexer)
    nn.Module.__init__(indexer)
    indexer.max_seqs = 4
    indexer.max_tokens = 8
    indexer._compressed_table_width = 2
    indexer._compressed_block_table = torch.full((4, 2), -1, dtype=torch.int32)
    indexer._sequence_lengths = torch.zeros(4, dtype=torch.int32)
    indexer._raw_state_slot_ids = torch.full((4,), -1, dtype=torch.int32)
    indexer._decode_query_start_loc = torch.zeros(5, dtype=torch.int32)
    indexer._prefill_query_start_loc = torch.zeros(5, dtype=torch.int32)
    indexer._prefill_request_ids = torch.empty(8, dtype=torch.int32)
    indexer._num_accepted_tokens = torch.ones(4, dtype=torch.int32)
    indexer._reset_mask = torch.zeros(4, dtype=torch.bool)
    indexer._prefix_lengths = torch.zeros(4, dtype=torch.int32)

    block_table = torch.tensor(
        [[10, 11], [20, 21], [30, 31], [-1, -1]], dtype=torch.int32
    )
    metadata = SimpleNamespace(
        num_reqs=4,
        block_table=block_table,
        seq_lens=torch.tensor([11, 19, 23, 0], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 2, 5, 8, 8], dtype=torch.int32),
        num_decodes=2,
        num_decode_tokens=5,
        req_id_per_token=torch.tensor([0, 0, 1, 1, 1, 2, 2, 2], dtype=torch.int32),
        selector_state_slot_ids=torch.tensor([7, 3, 5, -1], dtype=torch.int32),
        selector_state_is_fresh=torch.tensor([False, False, True, False]),
        selector_num_accepted_tokens=torch.ones(4, dtype=torch.int32),
    )

    indexer._stage_metadata(metadata, rows=8)
    decode_block_table = indexer._compressed_block_table.clone()
    decode_sequence_lengths = indexer._sequence_lengths.clone()
    decode_state_slots = indexer._raw_state_slot_ids.clone()
    prefill_request_ids = indexer._stage_prefill_metadata(
        metadata, num_decodes=2, decode_rows=5, rows=8
    )

    assert torch.equal(decode_block_table, block_table)
    assert torch.equal(
        decode_sequence_lengths, torch.tensor([11, 19, 23, 0], dtype=torch.int32)
    )
    assert torch.equal(
        decode_state_slots, torch.tensor([7, 3, 5, -1], dtype=torch.int32)
    )
    assert torch.equal(
        indexer._prefill_query_start_loc,
        torch.tensor([0, 3, 3, 3, 3], dtype=torch.int32),
    )
    assert torch.equal(prefill_request_ids, torch.tensor([0, 0, 0], dtype=torch.int32))
    assert torch.equal(
        indexer._compressed_block_table,
        torch.tensor([[30, 31], [-1, -1], [-1, -1], [-1, -1]], dtype=torch.int32),
    )
    assert torch.equal(
        indexer._sequence_lengths, torch.tensor([23, 0, 0, 0], dtype=torch.int32)
    )
    assert torch.equal(
        indexer._raw_state_slot_ids,
        torch.tensor([5, -1, -1, -1], dtype=torch.int32),
    )
