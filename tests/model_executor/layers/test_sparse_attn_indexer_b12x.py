# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
import types

import torch

from vllm.model_executor.layers import sparse_attn_indexer as indexer_mod


class _FakeWorkspaceManager:
    def __init__(self) -> None:
        self.specs: tuple[tuple[tuple[int, ...], torch.dtype], ...] | None = None

    def get_simultaneous(
        self, *shapes_and_dtypes: tuple[tuple[int, ...], torch.dtype]
    ) -> list[torch.Tensor]:
        self.specs = shapes_and_dtypes
        return [
            torch.empty(shape, dtype=dtype) for shape, dtype in shapes_and_dtypes
        ]


def _install_fake_b12x_indexer(monkeypatch, calls: list[tuple]):
    b12x_mod = types.ModuleType("b12x")
    attention_mod = types.ModuleType("b12x.attention")
    attention_indexer_mod = types.ModuleType("b12x.attention.indexer")
    kernel_mod = types.ModuleType("b12x.attention.indexer.kernel")
    tiled_topk_mod = types.ModuleType("b12x.attention.indexer.tiled_topk")
    integration_mod = types.ModuleType("b12x.integration")
    integration_indexer_mod = types.ModuleType("b12x.integration.indexer")

    def uses_paged_mqa_schedule(*, q_rows: int, max_pages: int) -> bool:
        calls.append(("schedule", q_rows, max_pages))
        return False

    def run_paged_windowed_tiled_logits_kernel(**kwargs):
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
        kwargs["output_values"].copy_(kwargs["candidate_values"][-1])
        kwargs["output_indices"].copy_(kwargs["candidate_indices"][-1])
        return kwargs["output_values"], kwargs["output_indices"]

    integration_indexer_mod.uses_paged_mqa_schedule = uses_paged_mqa_schedule
    kernel_mod.run_paged_windowed_tiled_logits_kernel = (
        run_paged_windowed_tiled_logits_kernel
    )
    tiled_topk_mod.run_tiled_topk = run_tiled_topk
    tiled_topk_mod.merge_tiled_topk_candidates = merge_tiled_topk_candidates

    monkeypatch.setitem(sys.modules, "b12x", b12x_mod)
    monkeypatch.setitem(sys.modules, "b12x.attention", attention_mod)
    monkeypatch.setitem(sys.modules, "b12x.attention.indexer", attention_indexer_mod)
    monkeypatch.setitem(sys.modules, "b12x.attention.indexer.kernel", kernel_mod)
    monkeypatch.setitem(
        sys.modules, "b12x.attention.indexer.tiled_topk", tiled_topk_mod
    )
    monkeypatch.setitem(sys.modules, "b12x.integration", integration_mod)
    monkeypatch.setitem(
        sys.modules, "b12x.integration.indexer", integration_indexer_mod
    )


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
