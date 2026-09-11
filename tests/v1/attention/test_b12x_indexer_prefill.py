# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefill plan capacity of the B12X sparse indexer, checked without CUDA.

Selection borrows plan scratch from the workspace reserved during profiling.
Every prefill chunk the metadata builder can emit under the batched-token
limit must resolve to a plan prepared at init; a chunk beyond the prepared
rows would compile a plan while serving and grow the workspace under the
captured decode graphs.
"""

import pytest
import torch

from vllm.v1.attention.backends.mla import b12x_indexer as indexer


def _chunk_rows(context, query_len=8192):
    builder = object.__new__(indexer.B12xIndexerMetadataBuilder)
    builder.max_prefill_buffer_size = 40 * 262144
    chunks = builder._split_prefill_chunks(
        torch.tensor([context], dtype=torch.int32),
        torch.tensor([query_len], dtype=torch.int32),
        0,
        512 * 1024 * 1024,
    )
    return [query.stop - query.start for _, query in chunks]


def _prepared_prefill_indexer(max_q_rows):
    obj = indexer.B12xSparseIndexer.__new__(indexer.B12xSparseIndexer)
    obj.topk_indices_buffer = torch.empty((1, 1), dtype=torch.int32)
    obj._prefill_plan_sizes = indexer._prefill_plan_row_counts(max_q_rows, False)
    obj._prefill_plans = {rows: f"prefill-{rows}" for rows in obj._prefill_plan_sizes}
    obj._decode_plan_sizes, obj._decode_plans = [], {}

    def compile_at_runtime(mode, q_rows):
        raise AssertionError(f"compiled a {mode} plan for {q_rows} rows while serving")

    obj._make_plan = compile_at_runtime
    return obj


def test_prefill_plan_rows_cover_the_batched_token_limit(monkeypatch):
    monkeypatch.setenv("VLLM_SPARSE_INDEXER_MAX_LOGITS_MB", "512")
    monkeypatch.delenv("B12X_PAGED_INDEX_SUPERTILE_K", raising=False)
    assert indexer._prefill_plan_row_counts(8192, False) == [4096, 8192]
    assert indexer._prefill_plan_row_counts(2048, False) == [2048]
    with_sort = indexer._prefill_plan_row_counts(8192, True)
    assert with_sort[-1] == 8192 and 4096 in with_sort


@pytest.mark.parametrize(
    "context",
    [
        8192,  # first chunk of a request: one piece of every query row
        16384,  # logits budget still allows the whole 8192-row chunk
        24576,  # odd 5461-row head piece above the logits-budget rows
    ],
)
def test_prefill_chunks_use_plans_prepared_before_capture(monkeypatch, context):
    """Every prefill chunk the builder emits resolves to a prepared plan.

    Selection borrows plan scratch from the workspace reserved during
    profiling; a chunk beyond the prepared rows would compile a plan and
    grow the workspace under the captured decode graphs.
    """
    monkeypatch.setenv("VLLM_SPARSE_INDEXER_MAX_LOGITS_MB", "512")
    monkeypatch.delenv("B12X_PAGED_INDEX_SUPERTILE_K", raising=False)
    rows = _chunk_rows(context)
    assert max(rows) > indexer._prefill_profile_q_rows(8192)
    obj = _prepared_prefill_indexer(8192)
    for chunk_rows in rows:
        plan = obj._get_plan("prefill", chunk_rows)
        assert plan in obj._prefill_plans.values()
    assert obj._prefill_plan_sizes == [4096, 8192]
