# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.v1.attention.backend import AttentionBackend, MultipleOf
from vllm.v1.attention.backends import b12x_attn
from vllm.v1.attention.backends.b12x_attn import (
    B12XPagedAttentionBackend,
    B12XPagedAttentionImpl,
    _max_page_table_width,
)
from vllm.v1.worker.utils import select_common_block_size


class _Page128OnlyBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "PAGE128_ONLY"

    @staticmethod
    def get_impl_cls():
        raise NotImplementedError

    @staticmethod
    def get_builder_cls():
        raise NotImplementedError

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [128]


def test_b12x_dense_backend_advertises_page128() -> None:
    assert B12XPagedAttentionBackend.get_supported_kernel_block_sizes() == [64, 128]
    assert B12XPagedAttentionBackend.supports_block_size(64)
    assert B12XPagedAttentionBackend.supports_block_size(128)
    assert not B12XPagedAttentionBackend.supports_block_size(32)
    assert B12XPagedAttentionBackend.get_preferred_block_size(16) == 128
    assert B12XPagedAttentionBackend.get_preferred_block_size(64) == 64


def test_b12x_dense_backend_advertises_sliding_window() -> None:
    assert B12XPagedAttentionBackend.supports_sliding_window()


def test_b12x_dense_kv_cache_shape_accepts_page128() -> None:
    assert B12XPagedAttentionBackend.get_kv_cache_shape(
        num_blocks=3,
        block_size=128,
        num_kv_heads=4,
        head_size=128,
        cache_dtype_str="bfloat16",
    ) == (3, 2, 128, 4, 128)


def test_b12x_dense_can_share_page128_group() -> None:
    assert (
        select_common_block_size(
            128,
            [B12XPagedAttentionBackend, _Page128OnlyBackend],
        )
        == 128
    )


def test_b12x_hybrid_align_reserves_expanded_page_table() -> None:
    assert _max_page_table_width(4096, 128, 4096, "none") == 32
    assert _max_page_table_width(4096, 128, 4096, "align") == 64

    storage_block_size = 3200
    expanded_width = (
        (4096 + storage_block_size - 1)
        // storage_block_size
        * (storage_block_size // 128)
    )
    assert expanded_width == 50
    assert expanded_width <= _max_page_table_width(4096, 128, 4096, "align")


def test_b12x_lazily_prepares_missing_decode_capture_bucket(monkeypatch) -> None:
    impl = object.__new__(B12XPagedAttentionImpl)
    plan = SimpleNamespace(layout=SimpleNamespace(nbytes=64))
    created: list[int] = []

    def create_plan(size: int) -> SimpleNamespace:
        created.append(size)
        return plan

    impl._decode_plans = {}
    impl._create_decode_plan = create_plan
    impl._scratch_nbytes = 64
    impl._extend_plan = object()
    metadata = SimpleNamespace(max_query_len=1)
    monkeypatch.setattr(b12x_attn, "_capture_alloc_forbidden", lambda: False)

    selected, fixed_split_size = impl._select_plan(metadata, 7, 7)

    assert selected is plan
    assert fixed_split_size is None
    assert impl._decode_plans == {7: plan}
    assert created == [7]

    assert impl._select_plan(metadata, 7, 7) == (plan, None)
    assert created == [7]


def test_b12x_missing_decode_bucket_fails_closed_during_capture(monkeypatch) -> None:
    impl = object.__new__(B12XPagedAttentionImpl)
    impl._decode_plans = {}
    impl._create_decode_plan = lambda size: pytest.fail(
        f"created plan for batch {size} during capture"
    )
    impl._scratch_nbytes = 64
    impl._extend_plan = object()
    metadata = SimpleNamespace(max_query_len=1)
    monkeypatch.setattr(b12x_attn, "_capture_alloc_forbidden", lambda: True)

    with pytest.raises(RuntimeError, match="batch size 7"):
        impl._select_plan(metadata, 7, 7)
