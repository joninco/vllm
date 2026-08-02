# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm.v1.attention.backends.mla.compressor_utils import get_c128a_topk_width

_MIB = 1 << 20
_SCRATCH_ALIGNMENT = 1024


def _load_b12x_module(monkeypatch) -> types.ModuleType:
    forward_context_module = types.ModuleType("vllm.forward_context")
    forward_context_module.get_forward_context = lambda: None  # type: ignore[attr-defined]

    common_ops_module = types.ModuleType("vllm.models.deepseek_v4.common.ops")
    common_ops_module.compute_dcp_global_topk_indices_and_lens = (  # type: ignore[attr-defined]
        lambda *args: None
    )
    common_ops_module.compute_global_topk_indices_and_lens = (  # type: ignore[attr-defined]
        lambda *args: None
    )

    flashmla_module = types.ModuleType("vllm.models.deepseek_v4.nvidia.flashmla")

    class _FlashMLABackend:
        pass

    class _FlashMLAAttention(torch.nn.Module):
        pass

    flashmla_module.DeepseekV4FlashMLAAttention = _FlashMLAAttention  # type: ignore[attr-defined]
    flashmla_module.DeepseekV4FlashMLABackend = _FlashMLABackend  # type: ignore[attr-defined]

    metadata_module = types.ModuleType("vllm.models.deepseek_v4.sparse_mla")
    metadata_module.DeepseekV4FlashMLAMetadata = type(  # type: ignore[attr-defined]
        "DeepseekV4FlashMLAMetadata", (), {}
    )

    compressor_module = types.ModuleType(
        "vllm.v1.attention.backends.mla.compressor_utils"
    )

    def fake_c128a_topk_width(max_model_len: int, compress_ratio: int) -> int:
        compressed_width = (max_model_len + compress_ratio - 1) // compress_ratio
        return (compressed_width + 127) // 128 * 128

    compressor_module.get_c128a_topk_width = fake_c128a_topk_width  # type: ignore[attr-defined]

    attention_ops_module = types.ModuleType("vllm.v1.attention.ops.common")
    attention_ops_module.cp_lse_ag_out_rs = lambda *args: None  # type: ignore[attr-defined]

    dcp_module = types.ModuleType("vllm.v1.attention.ops.dcp_alltoall")
    dcp_module.dcp_a2a_lse_reduce = lambda *args: None  # type: ignore[attr-defined]
    dcp_module.dcp_b12x_all_gather_heads = lambda *args: None  # type: ignore[attr-defined]

    workspace_module = types.ModuleType("vllm.v1.worker.workspace")
    workspace_module.current_workspace_manager = lambda: None  # type: ignore[attr-defined]

    for name, module in {
        "vllm.forward_context": forward_context_module,
        "vllm.models.deepseek_v4.common.ops": common_ops_module,
        "vllm.models.deepseek_v4.nvidia.flashmla": flashmla_module,
        "vllm.models.deepseek_v4.sparse_mla": metadata_module,
        "vllm.v1.attention.backends.mla.compressor_utils": compressor_module,
        "vllm.v1.attention.ops.common": attention_ops_module,
        "vllm.v1.attention.ops.dcp_alltoall": dcp_module,
        "vllm.v1.worker.workspace": workspace_module,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    source = Path(__file__).parents[3] / "vllm/models/deepseek_v4/nvidia/b12x.py"
    spec = importlib.util.spec_from_file_location("_test_dsv4_b12x", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def _align_scratch(value: int) -> int:
    return (value + _SCRATCH_ALIGNMENT - 1) & -_SCRATCH_ALIGNMENT


def _split_chunks_for_contract(
    *, rows: int, width: int, max_chunks: int | None = None
) -> int:
    rows = max(int(rows), 1)
    width = max(int(width), 1)
    chunk_limit = 256 if max_chunks is None else max(1, min(max_chunks, 256))
    decode_chunks = (width + 11) // 12
    if rows <= 256 and decode_chunks <= chunk_limit:
        return decode_chunks
    wide_decode_chunks = (width + 63) // 64
    if rows <= 256 and wide_decode_chunks <= chunk_limit:
        return wide_decode_chunks
    batched_chunks = (width + 1023) // 1024
    return batched_chunks if batched_chunks <= chunk_limit else chunk_limit


class _CompressedMLACaps:
    max_q_rows: int
    max_chunks_per_row: int
    max_q_chunks: int | None
    num_q_heads: int
    v_head_dim: int

    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)
        self.max_q_chunks = kwargs.get("max_q_chunks")


class _CompressedMLAPlan:
    def __init__(self, caps: _CompressedMLACaps) -> None:
        self.caps = caps

    def shapes_and_dtypes(
        self,
    ) -> tuple[tuple[tuple[int, ...], torch.dtype], ...]:
        caps = self.caps
        rows = max(int(caps.max_q_rows), 1)
        chunks = max(int(caps.max_chunks_per_row), 1)
        q_chunks = rows * chunks
        if caps.max_q_chunks is not None:
            q_chunks = max(q_chunks, int(caps.max_q_chunks))

        cursor = 0
        cursor = _align_scratch(cursor)
        cursor += q_chunks * caps.num_q_heads * caps.v_head_dim * 2
        cursor = _align_scratch(cursor)
        cursor += q_chunks * caps.num_q_heads * 4
        cursor = _align_scratch(cursor)
        cursor += rows * caps.num_q_heads * 4
        for _ in range(3):
            cursor = _align_scratch(cursor)
            cursor += 4
        cursor = _align_scratch(cursor)
        return (((max(cursor, _SCRATCH_ALIGNMENT),), torch.uint8),)


class _RecordingWorkspaceManager:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[tuple[int, ...], torch.dtype], ...]] = []

    def get_simultaneous(
        self, *shapes_and_dtypes: tuple[tuple[int, ...], torch.dtype]
    ) -> list[torch.Tensor]:
        self.calls.append(shapes_and_dtypes)
        return []


def _spec_bytes(
    specs: tuple[tuple[tuple[int, ...], torch.dtype], ...],
) -> int:
    total = 0
    for shape, dtype in specs:
        numel = 1
        for dim in shape:
            numel *= int(dim)
        total += _align_scratch(numel * dtype.itemsize)
    return total


def _install_fake_compressed_mla(monkeypatch) -> None:
    sparkinfer_package = types.ModuleType("sparkinfer")
    sparkinfer_package.__path__ = []
    attention_package = types.ModuleType("sparkinfer.attention")
    attention_package.__path__ = []
    compressed_mla_module = types.ModuleType("sparkinfer.attention.compressed_mla")

    compressed_mla_module.Caps = _CompressedMLACaps  # type: ignore[attr-defined]
    compressed_mla_module.plan = _CompressedMLAPlan  # type: ignore[attr-defined]
    compressed_mla_module.split_chunks_for_contract = (  # type: ignore[attr-defined]
        _split_chunks_for_contract
    )

    monkeypatch.setitem(sys.modules, "sparkinfer", sparkinfer_package)
    monkeypatch.setitem(sys.modules, "sparkinfer.attention", attention_package)
    monkeypatch.setitem(
        sys.modules,
        "sparkinfer.attention.compressed_mla",
        compressed_mla_module,
    )


def _runtime_scratch_bytes(*, rows: int, width: int, heads: int) -> int:
    max_chunks = (width + 63) // 64
    chunks = _split_chunks_for_contract(
        rows=rows,
        width=width,
        max_chunks=max_chunks,
    )
    plan = _CompressedMLAPlan(
        _CompressedMLACaps(
            device="cpu",
            num_q_heads=heads,
            max_q_rows=rows,
            max_width=width,
            head_dim=512,
            v_head_dim=512,
            page_size=64,
            max_chunks_per_row=chunks,
        )
    )
    return _spec_bytes(plan.shapes_and_dtypes())


@pytest.mark.parametrize(
    ("max_model_len", "expected_width"),
    [(327680, 2560), (524288, 4096)],
)
def test_c128a_topk_width_is_shared_and_aligned(
    max_model_len: int,
    expected_width: int,
) -> None:
    assert get_c128a_topk_width(max_model_len, 128) == expected_width


def _make_reserve_layer(
    b12x_mod,
    *,
    compress_ratio: int,
    max_model_len: int,
    max_num_batched_tokens: int,
    index_topk: int,
    window_size: int,
    page_size: int,
    dcp_world_size: int,
):
    layer = object.__new__(b12x_mod.DeepseekV4B12xMLAAttention)
    torch.nn.Module.__init__(layer)
    layer.compress_ratio = compress_ratio
    layer.topk_indices_buffer = (
        torch.empty((1, index_topk), dtype=torch.int32) if compress_ratio == 4 else None
    )
    layer.indexer = None
    layer.max_model_len = max_model_len
    layer.window_size = window_size
    layer.max_num_batched_tokens = max_num_batched_tokens
    layer.swa_cache_layer = SimpleNamespace(block_size=page_size)
    layer.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=dcp_world_size,
        )
    )
    return layer


def test_compressed_mla_reserve_covers_reporter_runtime_envelope(
    monkeypatch,
) -> None:
    b12x_mod = _load_b12x_module(monkeypatch)
    _install_fake_compressed_mla(monkeypatch)
    workspace_manager = _RecordingWorkspaceManager()
    monkeypatch.setattr(
        b12x_mod, "current_workspace_manager", lambda: workspace_manager
    )

    max_num_batched_tokens = 2048
    max_model_len = 524288
    max_num_seqs = 16
    speculative_tokens = 5
    tensor_parallel_size = 2
    model_heads = 64
    local_heads = model_heads // tensor_parallel_size
    window_size = 128
    index_topk = 512
    page_size = 64
    widths = {
        1: window_size,
        4: window_size + index_topk,
        128: window_size + ((max_model_len + 128 - 1) // 128 + 127) // 128 * 128,
    }
    expected_max_q_chunks = {1: 2048, 4: 2560, 128: 16896}
    expected_reserve_mib = {
        1: 64.5029296875,
        4: 80.5654296875,
        128: 530.3154296875,
    }
    assert max_num_seqs * (speculative_tokens + 1) == 96
    assert max_num_seqs * (speculative_tokens + 1) <= max_num_batched_tokens

    q = torch.empty((1, local_heads, 512), dtype=torch.bfloat16)
    reserve_bytes: dict[int, int] = {}
    for compress_ratio, width in widths.items():
        layer = _make_reserve_layer(
            b12x_mod,
            compress_ratio=compress_ratio,
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            index_topk=index_topk,
            window_size=window_size,
            page_size=page_size,
            dcp_world_size=1,
        )

        layer._reserve_dummy_compressed_mla_scratch(q)

        reserve_bytes[compress_ratio] = _spec_bytes(workspace_manager.calls[-1])
        split_cap = b12x_mod._compressed_mla_split_cap(width)
        max_q_chunks = b12x_mod._max_compressed_mla_q_chunks(
            max_num_batched_tokens,
            width,
            split_cap,
            _split_chunks_for_contract,
        )
        assert max_q_chunks == expected_max_q_chunks[compress_ratio]
        assert reserve_bytes[compress_ratio] / _MIB == pytest.approx(
            expected_reserve_mib[compress_ratio]
        )

    # Decode and extend use the same scratch planner. Sweep the shared row
    # envelope once; the decode envelope (1..96) is a strict subset.
    for compress_ratio, width in widths.items():
        runtime_bytes = max(
            _runtime_scratch_bytes(rows=rows, width=width, heads=local_heads)
            for rows in range(1, max_num_batched_tokens + 1)
        )
        assert reserve_bytes[compress_ratio] >= runtime_bytes, (
            f"compress_ratio={compress_ratio} "
            f"reserve={reserve_bytes[compress_ratio]} runtime={runtime_bytes}"
        )

    c128_width = widths[128]
    old_single_point_reserve = _runtime_scratch_bytes(
        rows=max_num_batched_tokens,
        width=c128_width,
        heads=local_heads,
    )
    reported_runtime = _runtime_scratch_bytes(
        rows=232,
        width=c128_width,
        heads=local_heads,
    )
    assert c128_width == 4224
    assert old_single_point_reserve / _MIB == pytest.approx(321.5029296875)
    assert reported_runtime / _MIB == pytest.approx(480.400390625)
    assert old_single_point_reserve < reported_runtime <= reserve_bytes[128]


def test_compressed_mla_reserve_uses_dcp_gathered_heads(monkeypatch) -> None:
    b12x_mod = _load_b12x_module(monkeypatch)
    _install_fake_compressed_mla(monkeypatch)
    workspace_manager = _RecordingWorkspaceManager()
    monkeypatch.setattr(
        b12x_mod, "current_workspace_manager", lambda: workspace_manager
    )

    max_rows = 2048
    width = 4224
    local_heads = 32
    dcp_world_size = 2
    layer = _make_reserve_layer(
        b12x_mod,
        compress_ratio=128,
        max_model_len=524288,
        max_num_batched_tokens=max_rows,
        index_topk=512,
        window_size=128,
        page_size=64,
        dcp_world_size=dcp_world_size,
    )

    q = torch.empty((1, local_heads, 512), dtype=torch.bfloat16)
    layer._reserve_dummy_compressed_mla_scratch(q)
    reserve_bytes = _spec_bytes(workspace_manager.calls[-1])
    runtime_bytes = max(
        _runtime_scratch_bytes(
            rows=rows,
            width=width,
            heads=local_heads * dcp_world_size,
        )
        for rows in range(1, max_rows + 1)
    )

    assert reserve_bytes / _MIB == pytest.approx(1060.6279296875)
    assert runtime_bytes / _MIB == pytest.approx(1060.1904296875)
    assert reserve_bytes >= runtime_bytes


def test_compressed_mla_q_chunk_envelope_is_cached(monkeypatch) -> None:
    b12x_mod = _load_b12x_module(monkeypatch)
    call_count = 0

    def counting_split_chunks_for_contract(**kwargs) -> int:
        nonlocal call_count
        call_count += 1
        return _split_chunks_for_contract(**kwargs)

    split_cap = b12x_mod._compressed_mla_split_cap(4224)
    for _ in range(2):
        assert (
            b12x_mod._max_compressed_mla_q_chunks(
                2048,
                4224,
                split_cap,
                counting_split_chunks_for_contract,
            )
            == 16896
        )

    assert call_count == 2048


def test_real_sparkinfer_planner_matches_pinned_workspace_envelope() -> None:
    compressed_mla = pytest.importorskip("sparkinfer.attention.compressed_mla")
    caps_cls = compressed_mla.Caps  # type: ignore[attr-defined]
    plan_scratch = compressed_mla.plan  # type: ignore[attr-defined]
    split_chunks = compressed_mla.split_chunks_for_contract  # type: ignore[attr-defined]

    max_rows = 2048
    heads = 32
    expected_max_q_chunks = {128: 2048, 640: 2560, 4224: 16896}
    expected_reserve_mib = {
        128: 64.5029296875,
        640: 80.5654296875,
        4224: 530.3154296875,
    }
    for width, expected_q_chunks in expected_max_q_chunks.items():
        split_cap = (width + 63) // 64
        max_q_chunks = max(
            rows
            * split_chunks(
                rows=rows,
                width=width,
                max_chunks=split_cap,
            )
            for rows in range(1, max_rows + 1)
        )
        assert max_q_chunks == expected_q_chunks

        reserve_splits = split_chunks(
            rows=max_rows,
            width=width,
            max_chunks=split_cap,
        )
        reserve_plan = plan_scratch(
            caps_cls(
                device="cpu",
                num_q_heads=heads,
                max_q_rows=max_rows,
                max_width=width,
                head_dim=512,
                v_head_dim=512,
                page_size=64,
                max_chunks_per_row=reserve_splits,
                max_q_chunks=max_q_chunks,
            )
        )
        reserve_bytes = _spec_bytes(reserve_plan.shapes_and_dtypes())
        assert reserve_bytes / _MIB == pytest.approx(expected_reserve_mib[width])

        runtime_bytes = 0
        for rows in range(1, max_rows + 1):
            runtime_splits = split_chunks(
                rows=rows,
                width=width,
                max_chunks=split_cap,
            )
            runtime_plan = plan_scratch(
                caps_cls(
                    device="cpu",
                    num_q_heads=heads,
                    max_q_rows=rows,
                    max_width=width,
                    head_dim=512,
                    v_head_dim=512,
                    page_size=64,
                    max_chunks_per_row=runtime_splits,
                )
            )
            runtime_bytes = max(
                runtime_bytes,
                _spec_bytes(runtime_plan.shapes_and_dtypes()),
            )
        assert reserve_bytes >= runtime_bytes
