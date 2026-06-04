# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
import types

import pytest
import torch

from vllm.model_executor.layers import sparse_attn_indexer as indexer_mod


class _FakeWorkspaceManager:
    def __init__(self, *, device: str | None = None) -> None:
        self.device = device
        self.specs: tuple[tuple[tuple[int, ...], torch.dtype], ...] | None = None

    def get_simultaneous(
        self, *shapes_and_dtypes: tuple[tuple[int, ...], torch.dtype]
    ) -> list[torch.Tensor]:
        self.specs = shapes_and_dtypes
        tensors = []
        for shape, dtype in shapes_and_dtypes:
            kwargs = {"dtype": dtype}
            if self.device is not None:
                kwargs["device"] = self.device
            tensors.append(torch.empty(shape, **kwargs))
        return tensors


def _install_fake_b12x_indexer(
    monkeypatch,
    calls: list[tuple],
    *,
    streaming_topk: bool = False,
    require_decode_tile_clear: bool = False,
):
    b12x_mod = types.ModuleType("b12x")
    attention_mod = types.ModuleType("b12x.attention")
    attention_indexer_mod = types.ModuleType("b12x.attention.indexer")
    extend_kernel_mod = types.ModuleType("b12x.attention.indexer.extend_kernel")
    kernel_mod = types.ModuleType("b12x.attention.indexer.kernel")
    tiled_topk_mod = types.ModuleType("b12x.attention.indexer.tiled_topk")
    integration_mod = types.ModuleType("b12x.integration")
    integration_indexer_mod = types.ModuleType("b12x.integration.indexer")

    class IndexerExtendMetadata:
        def __init__(self, *, k_start: torch.Tensor, k_end: torch.Tensor) -> None:
            self.k_start = k_start
            self.k_end = k_end

    def resolve_extend_prefill_block_k(**_kwargs) -> int:
        return 256

    def supports_extend_logits_kernel(**_kwargs):
        return True

    def run_extend_logits_kernel(**kwargs):
        calls.append(
            (
                "extend_score",
                kwargs["tile_k_offset"],
                kwargs["tile_num_k_tiles"],
                int(kwargs["tile_logits"].numel()),
            )
        )
        return kwargs["tile_logits"]

    def uses_paged_mqa_schedule(*, q_rows: int, max_pages: int) -> bool:
        calls.append(("schedule", q_rows, max_pages))
        return False

    def run_paged_windowed_tiled_logits_kernel(**kwargs):
        if require_decode_tile_clear:
            assert torch.isneginf(kwargs["tile_logits"]).all()
        calls.append(
            (
                "score",
                kwargs["source_page_offset"],
                kwargs["output_width_tokens"],
                int(kwargs["active_width"].item()),
                kwargs["stage_runtime_metadata"],
                kwargs["workspace"],
            )
        )
        return kwargs["tile_logits"]

    def run_tiled_topk(**kwargs):
        if streaming_topk:
            start = int(kwargs["input_index_offset"])
            extent = int(kwargs["input_extent"])
            topk = int(kwargs["topk"])
            local_indices = torch.arange(
                start + extent - 1,
                start + extent - 1 - topk,
                -1,
                dtype=torch.int32,
            )
            kwargs["output_indices"].copy_(
                local_indices.expand_as(kwargs["output_indices"])
            )
            kwargs["output_values"].copy_(
                local_indices.to(torch.float32).expand_as(kwargs["output_values"])
            )
            calls.append(
                (
                    "topk",
                    kwargs["input_index_offset"],
                    kwargs["input_extent"],
                    kwargs["num_k_tiles"],
                    kwargs.get("zero_row_start", False),
                )
            )
            return kwargs["output_values"], kwargs["output_indices"]

        calls.append(
            (
                "topk",
                kwargs["input_index_offset"],
                kwargs["input_extent"],
                kwargs["num_k_tiles"],
                kwargs["zero_row_start"],
            )
        )
        kwargs["output_values"].fill_(float(kwargs["input_index_offset"]))
        kwargs["output_indices"].fill_(int(kwargs["input_index_offset"]))
        return kwargs["output_values"], kwargs["output_indices"]

    def merge_tiled_topk_candidates(**kwargs):
        calls.append(("merge", tuple(kwargs["candidate_indices"].shape)))
        if streaming_topk:
            candidate_values = kwargs["candidate_values"]
            candidate_indices = kwargs["candidate_indices"]
            num_chunks, num_q_rows, topk = candidate_values.shape
            candidate_values_2d = candidate_values.permute(1, 0, 2).reshape(
                num_q_rows, num_chunks * topk
            )
            candidate_indices_2d = candidate_indices.permute(1, 0, 2).reshape(
                num_q_rows, num_chunks * topk
            )
            values, positions = torch.topk(
                candidate_values_2d,
                k=topk,
                dim=1,
                largest=True,
                sorted=False,
            )
            kwargs["output_values"].copy_(values)
            kwargs["output_indices"].copy_(
                torch.gather(candidate_indices_2d, 1, positions)
            )
            kwargs["merge_positions"].copy_(positions)
            return kwargs["output_values"], kwargs["output_indices"]

        kwargs["output_values"].copy_(kwargs["candidate_values"][-1])
        kwargs["output_indices"].copy_(kwargs["candidate_indices"][-1])
        return kwargs["output_values"], kwargs["output_indices"]

    def extend_tiled_topk(**_kwargs):
        calls.append(("extend_tiled_topk",))
        raise AssertionError("streaming wrapper must not call extend_tiled_topk")

    attention_indexer_mod.IndexerExtendMetadata = IndexerExtendMetadata
    attention_indexer_mod.resolve_extend_prefill_block_k = (
        resolve_extend_prefill_block_k
    )
    attention_indexer_mod.extend_tiled_topk = extend_tiled_topk
    extend_kernel_mod.run_extend_logits_kernel = run_extend_logits_kernel
    extend_kernel_mod.supports_extend_logits_kernel = supports_extend_logits_kernel
    integration_indexer_mod.uses_paged_mqa_schedule = uses_paged_mqa_schedule
    integration_indexer_mod.IndexerExtendMetadata = IndexerExtendMetadata
    integration_indexer_mod.resolve_extend_prefill_block_k = (
        resolve_extend_prefill_block_k
    )
    integration_indexer_mod.extend_tiled_topk = extend_tiled_topk
    kernel_mod.run_paged_windowed_tiled_logits_kernel = (
        run_paged_windowed_tiled_logits_kernel
    )
    tiled_topk_mod.run_tiled_topk = run_tiled_topk
    tiled_topk_mod.merge_tiled_topk_candidates = merge_tiled_topk_candidates

    monkeypatch.setitem(sys.modules, "b12x", b12x_mod)
    monkeypatch.setitem(sys.modules, "b12x.attention", attention_mod)
    monkeypatch.setitem(sys.modules, "b12x.attention.indexer", attention_indexer_mod)
    monkeypatch.setitem(
        sys.modules, "b12x.attention.indexer.extend_kernel", extend_kernel_mod
    )
    monkeypatch.setitem(sys.modules, "b12x.attention.indexer.kernel", kernel_mod)
    monkeypatch.setitem(
        sys.modules, "b12x.attention.indexer.tiled_topk", tiled_topk_mod
    )
    monkeypatch.setitem(sys.modules, "b12x.integration", integration_mod)
    monkeypatch.setitem(
        sys.modules, "b12x.integration.indexer", integration_indexer_mod
    )


def test_prefill_topk_normalization_converts_packed_indices_to_req_relative():
    chunk = types.SimpleNamespace(
        cu_seq_lens=torch.tensor([0, 3, 8], dtype=torch.int32),
        token_to_seq=torch.tensor([0, 0, 0, 1, 1, 1, 1, 1], dtype=torch.int32),
    )
    topk_indices = torch.tensor(
        [[0, 2, 3, -1], [4, 6, 7, 1]], dtype=torch.int32
    )

    indexer_mod._normalize_prefill_topk_to_req_relative(chunk, topk_indices)

    assert topk_indices.tolist() == [[0, 2, 0, -1], [1, 3, 4, 1]]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_prefill_topk_normalization_cuda_converts_in_place():
    chunk = types.SimpleNamespace(
        cu_seq_lens=torch.tensor([0, 3, 8], dtype=torch.int32, device="cuda"),
        token_to_seq=torch.tensor(
            [0, 0, 0, 1, 1, 1, 1, 1], dtype=torch.int32, device="cuda"
        ),
    )
    topk_indices = torch.tensor(
        [[0, 2, 3, -1], [4, 6, 7, 1]], dtype=torch.int32, device="cuda"
    )

    indexer_mod._normalize_prefill_topk_to_req_relative(chunk, topk_indices)
    torch.cuda.synchronize()

    assert topk_indices.cpu().tolist() == [[0, 2, 0, -1], [1, 3, 4, 1]]


def test_b12x_glm_decode_indexer_uses_paged_supertile_topk(monkeypatch):
    calls: list[tuple] = []
    _install_fake_b12x_indexer(monkeypatch, calls)
    workspace_manager = _FakeWorkspaceManager()
    monkeypatch.setattr(
        indexer_mod, "current_workspace_manager", lambda: workspace_manager
    )
    monkeypatch.setattr(indexer_mod, "_B12X_DECODE_TOPK_SUPERTILE_K", 512)

    q_rows = 2
    num_heads = 1
    topk = 4
    page_table_width = 10
    q_fp8 = torch.empty((q_rows, num_heads, 128), dtype=torch.uint8)
    weights = torch.empty((q_rows, num_heads, 1), dtype=torch.float32)
    kv_cache = torch.empty(
        (page_table_width, 64, 132), dtype=torch.uint8
    ).contiguous()
    seq_lens = torch.tensor([600, 640], dtype=torch.int32)
    block_table = torch.arange(
        q_rows * page_table_width, dtype=torch.int32
    ).reshape(q_rows, page_table_width)
    topk_indices = torch.empty((q_rows, topk), dtype=torch.int32)

    result = indexer_mod._run_b12x_decode_topk(
        q_fp8=q_fp8,
        weights=weights,
        kv_cache=kv_cache,
        seq_lens=seq_lens,
        block_table=block_table,
        schedule_metadata=None,
        topk_indices=topk_indices,
        topk_tokens=topk,
    )

    assert result is topk_indices
    assert topk_indices.tolist() == [[512] * topk, [512] * topk]
    assert workspace_manager.specs == (
        ((1,), torch.int32),
        ((32 * 512,), torch.float32),
        ((q_rows, topk), torch.float32),
        ((2, q_rows, topk), torch.float32),
        ((2, q_rows, topk), torch.int32),
        ((q_rows, topk), torch.int64),
    )
    assert calls == [
        ("schedule", q_rows, 8),
        ("score", 0, 512, 640, False, None),
        ("topk", 0, 512, 1, True),
        ("score", 8, 512, 640, False, None),
        ("topk", 512, 128, 1, True),
        ("merge", (2, q_rows, topk)),
    ]


def test_b12x_glm_decode_indexer_streams_multichunk_topk(monkeypatch):
    calls: list[tuple] = []
    _install_fake_b12x_indexer(
        monkeypatch,
        calls,
        streaming_topk=True,
        require_decode_tile_clear=True,
    )
    workspace_manager = _FakeWorkspaceManager()
    monkeypatch.setattr(
        indexer_mod, "current_workspace_manager", lambda: workspace_manager
    )
    monkeypatch.setattr(indexer_mod, "_B12X_DECODE_TOPK_SUPERTILE_K", 512)

    q_rows = 2
    num_heads = 1
    topk = 4
    page_table_width = 24
    q_fp8 = torch.empty((q_rows, num_heads, 128), dtype=torch.uint8)
    weights = torch.empty((q_rows, num_heads, 1), dtype=torch.float32)
    kv_cache = torch.empty(
        (page_table_width, 64, 132), dtype=torch.uint8
    ).contiguous()
    seq_lens = torch.full((q_rows,), 1536, dtype=torch.int32)
    block_table = torch.arange(
        q_rows * page_table_width, dtype=torch.int32
    ).reshape(q_rows, page_table_width)
    topk_indices = torch.empty((q_rows, topk), dtype=torch.int32)

    result = indexer_mod._run_b12x_decode_topk(
        q_fp8=q_fp8,
        weights=weights,
        kv_cache=kv_cache,
        seq_lens=seq_lens,
        block_table=block_table,
        schedule_metadata=None,
        topk_indices=topk_indices,
        topk_tokens=topk,
    )

    assert result is topk_indices
    assert workspace_manager.specs is not None
    assert workspace_manager.specs == (
        ((1,), torch.int32),
        ((32 * 512,), torch.float32),
        ((q_rows, topk), torch.float32),
        ((2, q_rows, topk), torch.float32),
        ((2, q_rows, topk), torch.int32),
        ((q_rows, topk), torch.int64),
    )
    assert [call for call in calls if call[0] == "topk"] == [
        ("topk", 0, 512, 1, True),
        ("topk", 512, 512, 1, True),
        ("topk", 1024, 512, 1, True),
    ]
    assert [call for call in calls if call[0] == "merge"] == [
        ("merge", (2, q_rows, topk)),
        ("merge", (2, q_rows, topk)),
    ]

    chunk_ranges = ((0, 512), (512, 512), (1024, 512))
    local_indices = torch.cat(
        [
            torch.arange(
                start + extent - 1,
                start + extent - 1 - topk,
                -1,
                dtype=torch.int32,
            )
            for start, extent in chunk_ranges
        ]
    )
    candidate_indices_ref = local_indices.expand(q_rows, -1)
    candidate_values_ref = candidate_indices_ref.to(torch.float32)
    _, positions = torch.topk(
        candidate_values_ref,
        k=topk,
        dim=1,
        largest=True,
        sorted=False,
    )
    expected_indices = torch.gather(candidate_indices_ref, 1, positions)
    assert torch.equal(topk_indices, expected_indices)


def test_b12x_glm_extend_indexer_streams_multichunk_topk(monkeypatch):
    calls: list[tuple] = []
    _install_fake_b12x_indexer(monkeypatch, calls, streaming_topk=True)
    workspace_manager = _FakeWorkspaceManager()
    monkeypatch.setattr(
        indexer_mod, "current_workspace_manager", lambda: workspace_manager
    )
    monkeypatch.setattr(indexer_mod, "_B12X_EXTEND_TOPK_SUPERTILE_K", 512)

    q_rows = 2
    num_heads = 1
    topk = 4
    total_seq_lens = 1280
    q_fp8 = torch.empty((q_rows, num_heads, 128), dtype=torch.uint8)
    weights = torch.empty((q_rows, num_heads), dtype=torch.float32)

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
    ) = indexer_mod._get_b12x_indexer_extend_buffers(
        q_fp8=q_fp8,
        topk_tokens=topk,
        total_seq_lens=total_seq_lens,
        head_dim=128,
        fp8_dtype=torch.uint8,
    )

    assert candidate_values.shape == (2, q_rows, topk)
    assert candidate_indices.shape == (2, q_rows, topk)
    assert workspace_manager.specs is not None
    assert workspace_manager.specs[6] == ((2, q_rows, topk), torch.float32)
    assert workspace_manager.specs[7] == ((2, q_rows, topk), torch.int32)

    result = indexer_mod._run_b12x_extend_tiled_topk_streaming(
        q_fp8=q_fp8,
        weights=weights,
        kv_fp8=(k_quant, k_scale.view(torch.float32).flatten()),
        metadata=types.SimpleNamespace(
            k_start=torch.zeros(q_rows, dtype=torch.int32),
            k_end=torch.full((q_rows,), total_seq_lens, dtype=torch.int32),
        ),
        topk=topk,
        contract_phantoms=None,
        workspace=None,
        tile_logits=tile_logits,
        lengths=lengths,
        output_values=topk_values,
        output_indices=topk_indices_out,
        candidate_values=candidate_values,
        candidate_indices=candidate_indices,
        merge_positions=merge_positions,
        supertile_k=512,
    )

    assert result is topk_indices_out
    assert not any(call[0] == "extend_tiled_topk" for call in calls)
    assert [call for call in calls if call[0] == "extend_score"] == [
        ("extend_score", 0, 2, int(tile_logits.numel())),
        ("extend_score", 2, 2, int(tile_logits.numel())),
        ("extend_score", 4, 1, int(tile_logits.numel())),
    ]
    assert [call for call in calls if call[0] == "topk"] == [
        ("topk", 0, 512, 2, False),
        ("topk", 512, 512, 2, False),
        ("topk", 1024, 256, 1, False),
    ]
    assert [call for call in calls if call[0] == "merge"] == [
        ("merge", (2, q_rows, topk)),
        ("merge", (2, q_rows, topk)),
    ]

    chunk_ranges = ((0, 512), (512, 512), (1024, 256))
    local_indices = torch.cat(
        [
            torch.arange(
                start + extent - 1,
                start + extent - 1 - topk,
                -1,
                dtype=torch.int32,
            )
            for start, extent in chunk_ranges
        ]
    )
    candidate_indices_ref = local_indices.expand(q_rows, -1)
    candidate_values_ref = candidate_indices_ref.to(torch.float32)
    _, positions = torch.topk(
        candidate_values_ref,
        k=topk,
        dim=1,
        largest=True,
        sorted=False,
    )
    expected_indices = torch.gather(candidate_indices_ref, 1, positions)
    assert torch.equal(topk_indices_out, expected_indices)


def test_b12x_extend_workspace_candidates_stay_fixed_for_1m_context(monkeypatch):
    calls: list[tuple] = []
    _install_fake_b12x_indexer(monkeypatch, calls)
    workspace_manager = _FakeWorkspaceManager(device="meta")
    monkeypatch.setattr(
        indexer_mod, "current_workspace_manager", lambda: workspace_manager
    )
    monkeypatch.setattr(indexer_mod, "_B12X_EXTEND_TOPK_SUPERTILE_K", 32768)

    q_rows = 2
    topk = 2048
    q_fp8 = torch.empty((q_rows, 1, 128), dtype=torch.uint8)

    indexer_mod._get_b12x_indexer_extend_buffers(
        q_fp8=q_fp8,
        topk_tokens=topk,
        total_seq_lens=1_048_576,
        head_dim=128,
        fp8_dtype=torch.uint8,
    )

    assert workspace_manager.specs is not None
    candidate_specs = [
        spec
        for spec in workspace_manager.specs
        if len(spec[0]) == 3 and spec[0][1:] == (q_rows, topk)
    ]
    assert candidate_specs == [
        ((2, q_rows, topk), torch.float32),
        ((2, q_rows, topk), torch.int32),
    ]
    assert all(spec[0] != (32, q_rows, topk) for spec in workspace_manager.specs)


def test_b12x_profile_skips_legacy_logits_dummy_allocation(monkeypatch):
    calls: list[tuple] = []
    _install_fake_b12x_indexer(monkeypatch, calls)
    workspace_manager = _FakeWorkspaceManager()
    monkeypatch.setattr(
        indexer_mod, "current_workspace_manager", lambda: workspace_manager
    )
    monkeypatch.setattr(
        indexer_mod,
        "get_forward_context",
        lambda: types.SimpleNamespace(attn_metadata=None),
    )
    monkeypatch.setattr(
        indexer_mod,
        "_ensure_b12x_sparse_indexer_supported",
        lambda: None,
    )
    monkeypatch.setattr(
        indexer_mod.current_platform,
        "fp8_dtype",
        lambda: torch.uint8,
        raising=False,
    )
    monkeypatch.setattr(indexer_mod.envs, "VLLM_SPARSE_INDEXER_MAX_LOGITS_MB", 1)

    q_rows = 2
    topk = 4
    total_seq_lens = 1024
    hidden_states = torch.empty((q_rows, 128), dtype=torch.bfloat16)
    kv_cache = torch.empty((1, 64, 132), dtype=torch.uint8)
    q_quant = torch.empty((q_rows, 1, 128), dtype=torch.uint8)
    k = torch.empty((total_seq_lens, 128), dtype=torch.uint8)
    weights = torch.empty((q_rows, 1), dtype=torch.float32)
    topk_indices_buffer = torch.empty((q_rows, topk), dtype=torch.int32)

    legacy_logits_elems = 1024 * 1024
    torch_empty = torch.empty

    def guarded_empty(*args, **kwargs):
        shape = args[0] if args else kwargs.get("size")
        if shape == legacy_logits_elems:
            raise AssertionError("B12X profile path allocated legacy logits dummy")
        if isinstance(shape, tuple) and shape == (legacy_logits_elems,):
            raise AssertionError("B12X profile path allocated legacy logits dummy")
        return torch_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", guarded_empty)

    result = indexer_mod.sparse_attn_indexer(
        hidden_states,
        "layers.0.attn",
        kv_cache,
        q_quant,
        None,
        k,
        weights,
        128,
        None,
        topk_tokens=topk,
        head_dim=128,
        max_model_len=total_seq_lens,
        total_seq_lens=total_seq_lens,
        topk_indices_buffer=topk_indices_buffer,
        skip_k_cache_insert=False,
        use_fp4_cache=False,
        use_b12x_sparse_indexer=True,
    )

    assert result is topk_indices_buffer
    assert workspace_manager.specs is not None
    assert ((2, q_rows, topk), torch.float32) in workspace_manager.specs
    assert ((2, q_rows, topk), torch.int32) in workspace_manager.specs


def test_b12x_extend_profile_rows_follow_logits_budget(monkeypatch):
    monkeypatch.setattr(indexer_mod.envs, "VLLM_SPARSE_INDEXER_MAX_LOGITS_MB", 512)

    assert (
        indexer_mod._get_b12x_indexer_extend_profile_q_rows(
            q_rows=65536,
            total_seq_lens=5_242_880,
        )
        == 25
    )

    assert (
        indexer_mod._get_b12x_indexer_extend_profile_q_rows(
            q_rows=65536,
            total_seq_lens=65_536,
        )
        == 2048
    )
