# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode context parallel exchange of the sparse attention path.

With decode context parallel and no prefill context parallel, every rank
attends over its KV shard for the query heads of the whole DCP group, so the
attention module gathers the query across the group before the sparse
attention op and combines the per-rank partial outputs by log-sum-exp
afterwards, as the generic MLA layer does. Without the exchange the partial
output carries ``dcp_world_size`` times the local heads and the value
projection cannot view it.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.models.deepseek_v32 import attention as attention_module
from vllm.models.deepseek_v32.attention import DeepseekV32Attention

_TOKENS = 3
_HEADS = 2
_LATENT = 4
_ROPE = 1
_V_HEAD = 5


def _module(events: list[str], *, dcp_world_size: int, full_ckv: bool):
    module = DeepseekV32Attention.__new__(DeepseekV32Attention)
    module.skip_topk = True
    module.indexer = None
    module.use_pcp = False
    module._vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=dcp_world_size, cp_kv_cache_interleave_size=1
        )
    )
    module._dense_mha_metadata_layer_name = "dense"
    module.layer_name = "layer"
    module._fp8_query = False
    module._fp8_kv_needs_view = False
    module._native_packed_kv_update = False
    module.num_heads = _HEADS
    module.num_local_heads = _HEADS
    module.kv_lora_rank = _LATENT
    module.v_head_dim = _V_HEAD
    module.W_UV = torch.ones(_HEADS, _LATENT, _V_HEAD)
    module.W_UK_T = torch.ones(_HEADS, 3, _LATENT)

    def forward_mqa(q, kv_cache, attn_metadata, layer):
        events.append(
            f"forward_mqa:{tuple(q.shape) if torch.is_tensor(q) else 'tuple'}"
        )
        heads = q.shape[1] if torch.is_tensor(q) else q[0].shape[1]
        rows = q.shape[0] if torch.is_tensor(q) else q[0].shape[0]
        return torch.ones(rows, heads, _LATENT), torch.zeros(rows, heads)

    module.impl = SimpleNamespace(
        dcp_world_size=dcp_world_size,
        pcp_world_size=1,
        uses_full_ckv_dcp=lambda attn_metadata, num_tokens: full_ckv,
        forward_mqa=forward_mqa,
    )

    def query_gather(q):
        events.append(f"query_gather:{tuple(q.shape)}")
        return q.repeat(1, dcp_world_size, 1)

    def combine(attn_out, lse, *, seq_lens, query_start_loc):
        events.append(
            f"combine:{tuple(attn_out.shape)}:{int(seq_lens.numel())}"
            f":{int(query_start_loc.numel())}"
        )
        return attn_out[:, :_HEADS]

    module.dcp_manager = (
        SimpleNamespace(query_gather=query_gather, combine=combine)
        if dcp_world_size > 1
        else None
    )
    return module


@pytest.mark.parametrize(
    ("dcp_world_size", "full_ckv", "expected"),
    [
        (
            4,
            False,
            [
                f"query_gather:{(_TOKENS, _HEADS, _LATENT + _ROPE)}",
                f"forward_mqa:{(_TOKENS, _HEADS * 4, _LATENT + _ROPE)}",
                f"combine:{(_TOKENS, _HEADS * 4, _LATENT)}:{_TOKENS}:{_TOKENS + 1}",
            ],
        ),
        (4, True, ["forward_mqa:tuple"]),
        (1, False, ["forward_mqa:tuple"]),
    ],
)
def test_dcp_decode_gathers_the_query_and_combines_the_partials(
    monkeypatch: pytest.MonkeyPatch,
    dcp_world_size: int,
    full_ckv: bool,
    expected: list[str],
) -> None:
    events: list[str] = []
    module = _module(events, dcp_world_size=dcp_world_size, full_ckv=full_ckv)
    if dcp_world_size == 1 or full_ckv:
        from vllm.v1.attention.ops import b12x_dcp

        monkeypatch.setattr(
            b12x_dcp,
            "active_dcp_transport",
            lambda: pytest.fail("DCP=1 or full-CKV prefill queried decode transport"),
        )
    attn_metadata = SimpleNamespace(
        num_actual_tokens=_TOKENS,
        num_decode_tokens=_TOKENS,
        num_decodes=_TOKENS,
        decode=None,
        seq_lens=torch.full((_TOKENS,), 7, dtype=torch.int32),
        query_start_loc=torch.arange(_TOKENS + 1, dtype=torch.int32),
        dcp_combine_seq_lens=torch.full((_TOKENS,), 2, dtype=torch.int32),
        dcp_combine_query_start_loc=torch.arange(_TOKENS + 1, dtype=torch.int32),
    )
    monkeypatch.setattr(
        attention_module,
        "get_attention_context",
        lambda layer_name: (attn_metadata, None, torch.zeros(1), None),
    )
    output = torch.zeros(_TOKENS, _HEADS * _V_HEAD)
    module._sparse_indexer_and_attn(
        torch.zeros(_TOKENS, 6),
        None,
        None,
        None,
        None,
        None,
        torch.zeros(_TOKENS, _HEADS, _LATENT),
        torch.zeros(_TOKENS, _HEADS, 3),
        torch.zeros(_TOKENS, _HEADS, _ROPE),
        output,
    )
    assert events == expected
    # Each output element is the sum of a ones row of the latent over W_UV.
    assert torch.equal(output, torch.full((_TOKENS, _HEADS * _V_HEAD), float(_LATENT)))


def _sharded_attention_reference(rows_per_request, interleave, device):
    """Independent causal attention over token-position shards and the full KV."""
    generator = torch.Generator(device=device).manual_seed(410)
    queries, partials, lses, expected, local_lengths = [], [], [], [], []
    for first_length in (1, 3, 64, 257, 4096):
        length = first_length + rows_per_request - 1
        q = torch.randn(
            rows_per_request,
            _HEADS * 4,
            _LATENT + _ROPE,
            dtype=torch.float64,
            device=device,
            generator=generator,
        )
        k = torch.randn(
            length,
            _LATENT + _ROPE,
            dtype=torch.float64,
            device=device,
            generator=generator,
        )
        v = torch.randn(
            length,
            _LATENT,
            dtype=torch.float64,
            device=device,
            generator=generator,
        )
        positions = torch.arange(length, device=device)
        visible = (
            positions[None, :]
            < (first_length + torch.arange(rows_per_request, device=device))[:, None]
        )
        scores = torch.einsum("rhd,td->rht", q, k) / (_LATENT + _ROPE) ** 0.5
        scores.masked_fill_(~visible[:, None, :], -float("inf"))
        expected.append(torch.softmax(scores, dim=-1) @ v)
        rank_out, rank_lse, rank_lens = [], [], []
        for rank in range(4):
            owned = (positions // interleave) % 4 == rank
            masked = scores.masked_fill(~owned[None, None, :], -float("inf"))
            lse = torch.logsumexp(masked, dim=-1)
            out = torch.nan_to_num(torch.softmax(masked, dim=-1)) @ v
            rank_out.append(out)
            rank_lse.append(lse)
            rank_lens.append((visible & owned).sum(-1).to(torch.int32))
        queries.append(q)
        partials.append(torch.stack(rank_out))
        lses.append(torch.stack(rank_lse))
        local_lengths.append(torch.stack(rank_lens))
    return (
        torch.cat(queries).float(),
        torch.cat(partials, dim=1).float(),
        torch.cat(lses, dim=1).float(),
        torch.cat(expected).float(),
        torch.cat(local_lengths, dim=1),
    )


class _ReferenceCollectives:
    """Emulate peer collectives while exercising the real local LSE correction."""

    world_size = 4

    def __init__(self, rank, partials, lses):
        self.rank_in_group = rank
        self.lses = lses
        self.corrected = partials * torch.softmax(lses, dim=0)[..., None]

    def all_gather(self, local_lse, dim):
        assert dim == 0
        # Empty rows are deliberately poisoned at the backend boundary. The
        # module's local lengths must cause the manager to mask every one.
        torch.testing.assert_close(local_lse, self.lses[self.rank_in_group])
        return self.lses.flatten(0, 1)

    def reduce_scatter(self, local_output, dim):
        assert dim == 1
        torch.testing.assert_close(
            local_output, self.corrected[self.rank_in_group], atol=2e-6, rtol=2e-5
        )
        terms = self.corrected.clone()
        terms[self.rank_in_group] = local_output
        first = self.rank_in_group * _HEADS
        return terms.sum(0)[:, first : first + _HEADS].contiguous()


def _through_mtp_block(monkeypatch, attend, rows, device):
    """Exercise the predictor and decoder forwards with neutral non-attention ops."""
    from vllm.models.deepseek_v32.nvidia import model as model_module
    from vllm.models.deepseek_v32.nvidia import mtp as mtp_module

    class SparseCall(torch.nn.Module):
        def forward(self, positions, hidden_states):
            return attend()

    block = model_module.DeepseekV32DecoderLayer.__new__(
        model_module.DeepseekV32DecoderLayer
    )
    torch.nn.Module.__init__(block)
    block.use_sequence_parallel = False
    block.input_layernorm = torch.nn.Identity()
    block.post_attention_layernorm = torch.nn.Identity()
    block.self_attn = SparseCall()
    block.mlp = torch.nn.Identity()
    monkeypatch.setattr(model_module.l2_prefetch, "issue", lambda *args: None)
    neutral_norm = lambda hidden, residual, norm: (hidden, residual)
    monkeypatch.setattr(model_module, "fused_allreduce_rms_norm", neutral_norm)
    monkeypatch.setattr(mtp_module, "fused_allreduce_rms_norm", neutral_norm)
    monkeypatch.setattr(mtp_module, "run_glm52_plan", lambda *args: None)
    monkeypatch.setattr(
        mtp_module,
        "fused_eh_norm",
        lambda positions, inputs_embeds, previous_hidden_states, *args: inputs_embeds,
    )
    layer = mtp_module.DeepseekV32MultiTokenPredictorLayer.__new__(
        mtp_module.DeepseekV32MultiTokenPredictorLayer
    )
    torch.nn.Module.__init__(layer)
    object.__setattr__(
        layer, "enorm", SimpleNamespace(weight=None, variance_epsilon=1e-6)
    )
    object.__setattr__(layer, "hnorm", SimpleNamespace(weight=None))
    object.__setattr__(layer, "shared_head", SimpleNamespace(norm=None))
    layer.eh_proj = torch.nn.Linear(
        _HEADS * _LATENT, _HEADS * _LATENT, bias=False, device=device
    )
    with torch.no_grad():
        layer.eh_proj.weight.copy_(torch.eye(_HEADS * _LATENT, device=device))
    layer._eh_plan = None
    layer.mtp_block = block
    hidden = torch.zeros(rows, _HEADS * _LATENT, device=device)
    result, recycled = layer(
        torch.zeros(rows, dtype=torch.long, device=device),
        torch.arange(rows, device=device),
        hidden,
        inputs_embeds=hidden,
    )
    torch.testing.assert_close(result, recycled)
    return result


@pytest.mark.parametrize("rows_per_request", [1, 4])
@pytest.mark.parametrize("interleave", [1, 4])
@pytest.mark.parametrize("drafter", [False, True])
@pytest.mark.parametrize("direct_dispatch", [False, True])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires one CUDA GPU")
def test_four_shard_combine_matches_causal_attention(
    monkeypatch, rows_per_request, interleave, drafter, direct_dispatch
):
    """Global lengths or request-wide draft masks must not admit empty partials."""
    import functools

    from vllm.v1.attention.ops.dcp import cp_lse_ag_out_rs

    device = torch.device("cuda")
    query, partials, lses, reference, local_lengths = _sharded_attention_reference(
        rows_per_request, interleave, device
    )
    rows = query.shape[0]
    for rank in range(4):
        module = _module([], dcp_world_size=4, full_ckv=False)
        module.v_head_dim = _LATENT
        module.W_UV = torch.eye(_LATENT, device=device).expand(_HEADS, -1, -1)
        group = _ReferenceCollectives(rank, partials, lses)
        first = rank * _HEADS
        local_query = query[:, first : first + _HEADS]

        def gather(q, local_query=local_query):
            torch.testing.assert_close(q, local_query)
            return query

        def backend(q, *args, rank=rank):
            torch.testing.assert_close(q, query)
            empty = local_lengths[rank] == 0
            output = partials[rank].clone()
            output[empty] = 999
            lse = lses[rank].clone()
            lse[empty] = 7
            return output, lse

        module.impl.forward_mqa = backend
        module.dcp_manager = SimpleNamespace(
            query_gather=gather,
            combine=functools.partial(cp_lse_ag_out_rs, cp_group=group),
        )
        boundaries = torch.arange(rows + 1, dtype=torch.int32, device=device)
        if direct_dispatch:
            from vllm.v1.attention.ops import b12x_dcp

            def direct_combine(
                output, lse, lengths, rank=rank, boundaries=boundaries, group=group
            ):
                assert lengths.data_ptr() == local_lengths[rank].data_ptr()
                return cp_lse_ag_out_rs(
                    output,
                    lse,
                    seq_lens=lengths,
                    query_start_loc=boundaries,
                    cp_group=group,
                )

            binding = SimpleNamespace(
                accepts=lambda *args: True, query=gather, combine=direct_combine
            )
            monkeypatch.setattr(
                b12x_dcp, "active_dcp_transport", lambda binding=binding: binding
            )
            module.dcp_manager = SimpleNamespace(
                query_gather=lambda *args: pytest.fail("Generic gather selected"),
                combine=lambda *args, **kwargs: pytest.fail("Generic combine selected"),
            )
        metadata = SimpleNamespace(
            num_actual_tokens=rows,
            dcp_combine_seq_lens=local_lengths[rank],
            dcp_combine_query_start_loc=boundaries,
        )
        monkeypatch.setattr(
            attention_module,
            "get_attention_context",
            lambda _, metadata=metadata: (metadata, None, None, None),
        )

        def attend(module=module, local_query=local_query):
            output = torch.empty(rows, _HEADS * _LATENT, device=device)
            module._sparse_indexer_and_attn(
                torch.empty(rows, 1, device=device),
                None,
                None,
                None,
                None,
                None,
                local_query[..., :_LATENT],
                torch.empty(rows, _HEADS, 3, device=device),
                local_query[..., _LATENT:],
                output,
            )
            return output

        actual = (
            _through_mtp_block(monkeypatch, attend, rows, device)
            if drafter
            else attend()
        )
        expected = reference[:, first : first + _HEADS].reshape(rows, -1)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("is_base_e", [False, True])
@pytest.mark.parametrize("dtype", [torch.float64, torch.bfloat16])
def test_projected_shard_merge_matches_global_attention(dtype, is_base_e):
    """Per-head value projection commutes with merging disjoint KV shards.

    CPU float64 attention supplies an independent oracle. BF16 tolerances
    apply only to these bounded fixtures, not serving numerical acceptance.
    """
    from vllm.v1.attention.ops.dcp import _lse_weighted_combine

    generator = torch.Generator().manual_seed(729)
    ranks, rows, heads, tokens, latent, value_dim = 4, 7, 8, 13, 9, 5
    scores = torch.randn(rows, heads, tokens, generator=generator, dtype=torch.float64)
    values = torch.randn(tokens, latent, generator=generator, dtype=torch.float64)
    weights = (
        torch.randn(heads, latent, value_dim, generator=generator, dtype=torch.float64)
        / latent**0.5
    )
    # Includes an all-empty padding row, causal prefixes and uneven shard tails.
    lengths = torch.tensor([0, 1, 2, 4, 5, 9, tokens])
    positions = torch.arange(tokens)
    visible = positions[None, :] < lengths[:, None]
    masked_scores = scores.masked_fill(~visible[:, None, :], -float("inf"))
    probabilities = torch.nan_to_num(torch.softmax(masked_scores, dim=-1))
    expected = torch.einsum("bht,tl,hlv->bhv", probabilities, values, weights)
    partials, lses = [], []
    for rank in range(ranks):
        owned = (positions // 2) % ranks == rank
        local_scores = masked_scores.masked_fill(~owned[None, None, :], -float("inf"))
        local_lse = torch.logsumexp(local_scores, dim=-1)
        # Empty backend outputs are undefined; the merge must mask NaN and inf.
        local_output = torch.softmax(local_scores, dim=-1) @ values
        local_output[~torch.isfinite(local_lse)] = (
            float("nan") if rank % 2 else float("inf")
        )
        partials.append(local_output)
        lses.append(local_lse)
    partials = torch.stack(partials).to(dtype)
    lses = torch.stack(lses)
    if not is_base_e:
        lses = lses / torch.log(torch.tensor(2.0, dtype=torch.float64))
    lses = lses.to(torch.float64 if dtype == torch.float64 else torch.float32)
    weights = weights.to(dtype)
    projected = torch.einsum("rbhl,hlv->rbhv", partials, weights)
    project_then_merge = _lse_weighted_combine(
        projected, lses, is_lse_base_on_e=is_base_e
    ).to(dtype)
    merged = _lse_weighted_combine(partials, lses, is_lse_base_on_e=is_base_e).to(dtype)
    merge_then_project = torch.einsum("bhl,hlv->bhv", merged, weights)
    tolerance = 1e-12 if dtype == torch.float64 else 0.025
    for actual in (project_then_merge, merge_then_project):
        assert torch.isfinite(actual).all()
        assert torch.count_nonzero(actual[0]) == 0
        torch.testing.assert_close(actual.double(), expected, atol=tolerance, rtol=0)
    torch.testing.assert_close(
        project_then_merge, merge_then_project, atol=tolerance, rtol=0
    )
