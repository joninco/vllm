# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Draft-step metadata policy of the B12X sparse MLA builder under DCP.

At DCP=1 every step-dependent length the decode kernels read is a view of
the shared sequence lengths, which the speculator advances in place. Under
DCP the builder localizes per-token lengths from the positions into its own
buffer at build time, so an in-place refresh of the per-request lengths would
leave the bound lengths stale. The builder therefore advertises in-place
draft updates only at DCP=1; under DCP the speculator rebuilds per step.
"""

from types import SimpleNamespace

import pytest
import torch

import vllm.v1.attention.backends.mla.b12x_mla_sparse as sparse


def _vllm_config(dcp_world_size: int) -> SimpleNamespace:
    parallel_config = SimpleNamespace(
        decode_context_parallel_size=dcp_world_size, enable_dbo=False
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(model_type="glm_moe_dsa"),
            get_num_attention_heads=lambda parallel: 8,
        ),
        parallel_config=parallel_config,
        scheduler_config=SimpleNamespace(max_num_batched_tokens=64, max_num_seqs=4),
        speculative_config=SimpleNamespace(
            num_speculative_tokens=3, parallel_drafting=False
        ),
    )


@pytest.mark.parametrize("dcp_world_size", [1, 4])
def test_sparse_builder_refreshes_draft_metadata_in_place_only_at_dcp1(
    monkeypatch, dcp_world_size
):
    def base_init(self, kv_cache_spec, layer_names, vllm_config, device):
        self.dcp_world_size = dcp_world_size
        self.device = device

    monkeypatch.setattr(sparse.SparseMLACommonMetadataBuilder, "__init__", base_init)
    monkeypatch.setattr(
        sparse.B12xMLASparseMetadataBuilder,
        "_init_reorder_batch_threshold",
        lambda self, threshold, **kwargs: None,
    )
    monkeypatch.setattr(
        sparse, "get_dcp_group", lambda: SimpleNamespace(rank_in_group=1)
    )

    builder = sparse.B12xMLASparseMetadataBuilder(
        SimpleNamespace(), ["layer"], _vllm_config(dcp_world_size), torch.device("cpu")
    )

    assert builder.supports_draft_decode_metadata_update is (dcp_world_size == 1)
