# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.mla import b12x_mla
from vllm.v1.attention.backends.mla.b12x_mla import (
    B12xMLABackend,
    B12xMLAImpl,
    B12xMLAMetadataBuilder,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum


def test_b12x_mla_is_registered_with_k3_envelope() -> None:
    assert AttentionBackendEnum.B12X_MLA.get_class() is B12xMLABackend
    assert B12xMLABackend.get_name() == "B12X_MLA"
    assert B12xMLABackend.get_supported_head_sizes() == [576]
    assert B12xMLABackend.supports_block_size(944)
    assert not B12xMLABackend.supports_block_size(936)
    assert B12xMLABackend.supports_compute_capability(DeviceCapability(12, 0))
    assert not B12xMLABackend.supports_compute_capability(DeviceCapability(10, 0))
    assert B12xMLABackend.supports_non_causal()
    assert (
        B12xMLAMetadataBuilder._cudagraph_support
        is b12x_mla.AttentionCGSupport.UNIFORM_BATCH
    )
    assert B12xMLAMetadataBuilder.query_len_support is b12x_mla.QueryLenSupport.UNIFORM


class _FakePlan:
    def shapes_and_dtypes(self):
        return (((256,), torch.uint8),)


class _FakeDenseMLA:
    def __init__(self) -> None:
        self.bindings: list[SimpleNamespace] = []
        self.compile_count = 0

    def bind(self, plan, **kwargs):
        binding = SimpleNamespace(plan=plan, **kwargs)
        self.bindings.append(binding)
        return binding

    def compile(self, *, binding) -> None:
        self.compile_count += 1

    def run(self, *, binding):
        lse = torch.zeros(
            binding.output.shape[:2],
            dtype=torch.float32,
            device=binding.output.device,
        )
        return binding.output, lse


def _fake_impl(monkeypatch, *, num_heads: int = 8) -> tuple[B12xMLAImpl, _FakeDenseMLA]:
    monkeypatch.setattr(b12x_mla, "is_workspace_manager_initialized", lambda: False)
    impl = object.__new__(B12xMLAImpl)
    impl.num_heads = num_heads
    impl.kv_lora_rank = 512
    impl.scale = 192**-0.5
    impl._compiled_bindings = set()
    impl._fallback_scratch = {}
    dense_mla = _FakeDenseMLA()
    impl._dense_mla = dense_mla
    return impl, dense_mla


def test_b12x_mla_adapter_binds_common_decode_metadata(monkeypatch) -> None:
    impl, dense_mla = _fake_impl(monkeypatch)
    batch = 2
    q_nope = torch.randn(batch, 8, 512, dtype=torch.bfloat16)
    q_rope = torch.randn(batch, 8, 64, dtype=torch.bfloat16)
    cache = torch.randn(4, 16, 576, dtype=torch.bfloat16)
    metadata = SimpleNamespace(
        dense_mla_plan=_FakePlan(),
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        decode=SimpleNamespace(
            block_table=torch.tensor([[0, 1], [2, 3]], dtype=torch.int32),
            seq_lens=torch.tensor([16, 32], dtype=torch.int32),
        ),
    )
    layer = SimpleNamespace(
        _q_scale=torch.tensor(0.25),
        _k_scale=torch.tensor(0.5),
    )

    output, lse = impl.forward_mqa((q_nope, q_rope), cache, metadata, layer)
    output_2, _ = impl.forward_mqa((q_nope, q_rope), cache, metadata, layer)

    assert output.shape == (batch, 8, 512)
    assert output.dtype == torch.bfloat16
    assert lse is not None and lse.dtype == torch.float32
    assert output_2.shape == output.shape
    assert dense_mla.compile_count == 1
    binding = dense_mla.bindings[0]
    assert binding.q.shape == (batch, 8, 576)
    assert binding.q.is_contiguous()
    assert binding.kv_cache is cache
    assert binding.page_table is metadata.decode.block_table
    assert binding.cache_seqlens is metadata.decode.seq_lens
    assert binding.q_scale is None
    assert binding.kv_scale is None
    assert binding.sm_scale == impl.scale


def test_b12x_mla_adapter_accepts_non_multiple_of_eight_heads(monkeypatch) -> None:
    impl, dense_mla = _fake_impl(monkeypatch, num_heads=6)
    q = torch.randn(1, 6, 576, dtype=torch.bfloat16)
    cache = torch.randn(2, 16, 576, dtype=torch.bfloat16)
    metadata = SimpleNamespace(
        dense_mla_plan=_FakePlan(),
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        decode=SimpleNamespace(
            block_table=torch.tensor([[0, 1]], dtype=torch.int32),
            seq_lens=torch.tensor([17], dtype=torch.int32),
        ),
    )
    layer = SimpleNamespace(_q_scale=torch.tensor(0.25), _k_scale=torch.tensor(0.5))

    output, _ = impl.forward_mqa(q, cache, metadata, layer)

    assert output.shape == (1, 6, 512)
    assert dense_mla.bindings[0].q.shape == (1, 6, 576)


def test_b12x_mla_adapter_binds_causal_multiquery_blocks(monkeypatch) -> None:
    impl, dense_mla = _fake_impl(monkeypatch)
    batch = 2
    query_len = 8
    total_q = batch * query_len
    q = torch.randn(total_q, 8, 576, dtype=torch.bfloat16)
    cache = torch.randn(8, 16, 576, dtype=torch.bfloat16)
    query_start_loc = torch.tensor([0, 8, 16], dtype=torch.int32)
    metadata = SimpleNamespace(
        dense_mla_plan=_FakePlan(),
        query_start_loc=query_start_loc,
        decode=SimpleNamespace(
            block_table=torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.int32),
            seq_lens=torch.tensor([32, 48], dtype=torch.int32),
        ),
    )
    layer = SimpleNamespace(
        _q_scale=torch.tensor(0.25),
        _k_scale=torch.tensor(0.5),
    )

    output, lse = impl.forward_mqa(q, cache, metadata, layer)

    binding = dense_mla.bindings[0]
    assert output.shape == (total_q, 8, 512)
    assert lse is not None and lse.shape == (total_q, 8)
    assert binding.q.shape[0] == total_q
    assert binding.cache_seqlens.shape[0] == batch
    assert binding.cu_seqlens_q.data_ptr() == query_start_loc.data_ptr()


def test_b12x_mla_builder_flattens_non_causal_draft_block(monkeypatch) -> None:
    builder = object.__new__(B12xMLAMetadataBuilder)
    builder._dense_mla_plan = _FakePlan()
    builder._max_dense_mla_rows = 16
    builder._dense_mla_flat_block_table = torch.zeros(16, 4, dtype=torch.int32)
    builder._dense_mla_flat_seq_lens = torch.empty(16, dtype=torch.int32)
    builder._dense_mla_flat_query_start_loc = torch.arange(17, dtype=torch.int32)

    source_table = torch.tensor([[3, 4, 5, 6]], dtype=torch.int32)
    metadata = SimpleNamespace(
        causal=False,
        num_decodes=1,
        num_decode_tokens=8,
        decode=SimpleNamespace(
            block_table=source_table,
            seq_lens=torch.tensor([32], dtype=torch.int32),
        ),
    )
    monkeypatch.setattr(
        b12x_mla.MLACommonMetadataBuilder,
        "build",
        lambda *args, **kwargs: metadata,
    )

    result = builder.build(0, SimpleNamespace())

    torch.testing.assert_close(
        result.dense_mla_flat_block_table,
        source_table.expand(8, -1),
    )
    torch.testing.assert_close(
        result.dense_mla_flat_seq_lens,
        torch.full((8,), 32, dtype=torch.int32),
    )
    torch.testing.assert_close(
        result.dense_mla_flat_query_start_loc,
        torch.arange(9, dtype=torch.int32),
    )


def test_b12x_mla_adapter_uses_flattened_non_causal_rows(monkeypatch) -> None:
    impl, dense_mla = _fake_impl(monkeypatch, num_heads=6)
    query_rows = 8
    q = torch.randn(query_rows, 6, 576, dtype=torch.bfloat16)
    cache = torch.randn(4, 16, 576, dtype=torch.bfloat16)
    source_table = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32)
    flat_table = source_table.expand(query_rows, -1).contiguous()
    flat_lens = torch.full((query_rows,), 49, dtype=torch.int32)
    flat_query_start = torch.arange(query_rows + 1, dtype=torch.int32)
    metadata = SimpleNamespace(
        dense_mla_plan=_FakePlan(),
        dense_mla_flat_block_table=flat_table,
        dense_mla_flat_seq_lens=flat_lens,
        dense_mla_flat_query_start_loc=flat_query_start,
        query_start_loc=torch.tensor([0, query_rows], dtype=torch.int32),
        decode=SimpleNamespace(
            block_table=source_table,
            seq_lens=torch.tensor([49], dtype=torch.int32),
        ),
    )
    layer = SimpleNamespace(_q_scale=torch.tensor(0.25), _k_scale=torch.tensor(0.5))

    output, lse = impl.forward_mqa(q, cache, metadata, layer)

    binding = dense_mla.bindings[0]
    assert binding.page_table is flat_table
    assert binding.cache_seqlens is flat_lens
    assert binding.cu_seqlens_q.data_ptr() == flat_query_start.data_ptr()
    assert output.shape == (query_rows, 6, 512)
    assert lse is not None and lse.shape == (query_rows, 6)


def test_b12x_mla_adapter_passes_fp8_scales(monkeypatch) -> None:
    impl, dense_mla = _fake_impl(monkeypatch)
    q = torch.empty(1, 8, 576, dtype=torch.float8_e4m3fn)
    cache = torch.empty(2, 16, 576, dtype=torch.float8_e4m3fn)
    metadata = SimpleNamespace(
        dense_mla_plan=_FakePlan(),
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        decode=SimpleNamespace(
            block_table=torch.tensor([[0, 1]], dtype=torch.int32),
            seq_lens=torch.tensor([17], dtype=torch.int32),
        ),
    )
    layer = SimpleNamespace(
        _q_scale=torch.tensor(0.25),
        _k_scale=torch.tensor(0.5),
    )

    impl.forward_mqa(q, cache, metadata, layer)

    binding = dense_mla.bindings[0]
    assert binding.q_scale is layer._q_scale
    assert binding.kv_scale is layer._k_scale
