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
):
    b12x_mod = types.ModuleType("b12x")
    attention_mod = types.ModuleType("b12x.attention")
    attention_indexer_mod = types.ModuleType("b12x.attention.indexer")
    integration_mod = types.ModuleType("b12x.integration")
    integration_mod.__path__ = []
    compressed_indexer_mod = types.ModuleType("b12x.integration.compressed_indexer")
    integration_indexer_mod = types.ModuleType("b12x.integration.indexer")

    class _Caps:
        def __init__(self, **kwargs) -> None:
            self.__dict__.update(kwargs)

    class _Plan:
        def __init__(self, specs, bind_impl) -> None:
            self._specs = specs
            self._bind_impl = bind_impl

        def shapes_and_dtypes(self):
            return self._specs

        def bind(self, scratch, **kwargs):
            return self._bind_impl(scratch, kwargs)

    def plan_compressed_indexer_scratch(caps):
        calls.append(
            (
                "compressed_plan",
                caps.max_q_rows,
                caps.max_page_table_width,
                caps.topk,
                caps.reserve_paged_logits,
            )
        )
        specs = (
            ((caps.max_q_rows, caps.topk), torch.int32),
            ((caps.max_q_rows,), torch.int32),
        )

        def bind_impl(scratch, kwargs):
            calls.append(
                (
                    "compressed_bind",
                    tuple(kwargs["real_page_table"].shape),
                    tuple(kwargs["cache_seqlens_int32"].shape),
                    kwargs["expected_num_q_heads"],
                )
            )
            return types.SimpleNamespace(scratch=scratch, **kwargs)

        return _Plan(specs, bind_impl)

    def index_topk_fp8(**kwargs):
        calls.append(
            (
                "compressed_index_topk",
                tuple(kwargs["q_fp8"].shape),
                tuple(kwargs["index_k_cache"].shape),
                kwargs["page_size"],
                kwargs["expected_num_q_heads"],
            )
        )
        kwargs["out_indices"].fill_(123)
        return kwargs["out_indices"]

    def plan_indexer_extend_scratch(caps):
        calls.append(
            (
                "extend_plan",
                caps.max_q_rows,
                caps.max_k_rows,
                caps.topk,
                caps.supertile_k,
                caps.prefill_block_k,
            )
        )
        specs = (
            ((caps.max_k_rows, 128), caps.k_dtype),
            ((caps.max_k_rows, 4), torch.uint8),
            ((caps.max_q_rows,), torch.int32),
            ((caps.max_q_rows,), torch.int32),
        )

        def bind_impl(scratch, kwargs):
            calls.append(
                (
                    "extend_bind",
                    kwargs["gather_rows"],
                    kwargs["topk"],
                    kwargs["k_start"] is not None,
                    kwargs["k_end"] is not None,
                )
            )
            scratch_views = types.SimpleNamespace(
                k_quant=scratch[0],
                k_scale_bytes=scratch[1],
                k_scale=scratch[1].view(torch.float32),
                metadata_k_start=scratch[2],
                metadata_k_end=scratch[3],
            )
            return types.SimpleNamespace(scratch=scratch_views, **kwargs)

        return _Plan(specs, bind_impl)

    def extend_tiled_topk(**kwargs):
        binding = kwargs["binding"]
        calls.append(
            (
                "extend_tiled_topk",
                tuple(kwargs["q_fp8"].shape),
                binding.gather_rows,
                binding.topk,
            )
        )
        output = torch.empty(
            (kwargs["q_fp8"].shape[0], binding.topk),
            dtype=torch.int32,
            device=kwargs["q_fp8"].device,
        )
        output.fill_(7)
        return output

    compressed_indexer_mod.COMPRESSED_INDEX_PAGE_SIZE = 64
    compressed_indexer_mod.B12XCompressedIndexerScratchCaps = _Caps
    compressed_indexer_mod.plan_compressed_indexer_scratch = (
        plan_compressed_indexer_scratch
    )
    compressed_indexer_mod.index_topk_fp8 = index_topk_fp8
    integration_mod.B12XIndexerExtendScratchCaps = _Caps
    integration_mod.plan_indexer_extend_scratch = plan_indexer_extend_scratch
    integration_indexer_mod.extend_tiled_topk = extend_tiled_topk

    monkeypatch.setitem(sys.modules, "b12x", b12x_mod)
    monkeypatch.setitem(sys.modules, "b12x.attention", attention_mod)
    monkeypatch.setitem(sys.modules, "b12x.attention.indexer", attention_indexer_mod)
    monkeypatch.setitem(
        sys.modules, "b12x.integration.compressed_indexer", compressed_indexer_mod
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


def test_b12x_glm_decode_indexer_uses_compressed_indexer_plan(monkeypatch):
    calls: list[tuple] = []
    _install_fake_b12x_indexer(monkeypatch, calls)
    workspace_manager = _FakeWorkspaceManager()
    monkeypatch.setattr(
        indexer_mod, "current_workspace_manager", lambda: workspace_manager
    )

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
    assert topk_indices.tolist() == [[123] * topk, [123] * topk]
    assert workspace_manager.specs == (
        ((q_rows, topk), torch.int32),
        ((q_rows,), torch.int32),
    )
    assert calls == [
        ("compressed_plan", q_rows, page_table_width, topk, False),
        ("compressed_bind", tuple(block_table.shape), tuple(seq_lens.shape), num_heads),
        (
            "compressed_index_topk",
            tuple(q_fp8.shape),
            (page_table_width, 64 * 132),
            64,
            num_heads,
        ),
    ]


def test_b12x_extend_binding_uses_plan_scratch(monkeypatch):
    calls: list[tuple] = []
    _install_fake_b12x_indexer(monkeypatch, calls)
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
    k_start = torch.zeros(q_rows, dtype=torch.int32)
    k_end = torch.full((q_rows,), total_seq_lens, dtype=torch.int32)

    binding = indexer_mod._get_b12x_indexer_extend_binding(
        q_fp8=q_fp8,
        topk_tokens=topk,
        total_seq_lens=total_seq_lens,
        fp8_dtype=torch.uint8,
        k_start=k_start,
        k_end=k_end,
    )

    assert binding.scratch.k_quant.shape == (total_seq_lens, 128)
    assert binding.scratch.k_scale_bytes.shape == (total_seq_lens, 4)
    assert binding.k_start is k_start
    assert binding.k_end is k_end
    assert workspace_manager.specs is not None
    assert workspace_manager.specs == (
        ((total_seq_lens, 128), torch.uint8),
        ((total_seq_lens, 4), torch.uint8),
        ((q_rows,), torch.int32),
        ((q_rows,), torch.int32),
    )
    assert calls == [
        ("extend_plan", q_rows, total_seq_lens, topk, 512, 256),
        ("extend_bind", total_seq_lens, topk, True, True),
    ]


def test_b12x_extend_warmup_uses_bound_scratch(monkeypatch):
    calls: list[tuple] = []
    _install_fake_b12x_indexer(monkeypatch, calls)
    workspace_manager = _FakeWorkspaceManager()
    monkeypatch.setattr(
        indexer_mod, "current_workspace_manager", lambda: workspace_manager
    )
    monkeypatch.setattr(indexer_mod, "_B12X_EXTEND_TOPK_SUPERTILE_K", 512)

    q_rows = 2
    topk = 4
    total_seq_lens = 1280
    q_fp8 = torch.empty((q_rows, 1, 128), dtype=torch.uint8)
    weights = torch.empty((q_rows, 1), dtype=torch.float32)

    binding = indexer_mod._get_b12x_indexer_extend_binding(
        q_fp8=q_fp8,
        topk_tokens=topk,
        total_seq_lens=total_seq_lens,
        fp8_dtype=torch.uint8,
    )

    indexer_mod._warmup_b12x_extend_indexer(
        q_fp8=q_fp8,
        weights=weights,
        binding=binding,
        topk_tokens=topk,
        total_seq_lens=total_seq_lens,
    )

    assert ("extend_tiled_topk", tuple(q_fp8.shape), total_seq_lens, topk) in calls


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
    assert ((total_seq_lens, 128), torch.uint8) in workspace_manager.specs
    assert ((total_seq_lens, 4), torch.uint8) in workspace_manager.specs
    assert any(call[0] == "extend_tiled_topk" for call in calls)


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
