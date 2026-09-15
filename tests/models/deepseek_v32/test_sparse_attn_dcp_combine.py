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

from vllm.envs import disable_envs_cache
from vllm.models.deepseek_v32 import attention as attention_module
from vllm.models.deepseek_v32.attention import DeepseekV32Attention


def _override_envs(monkeypatch, name, value):
    """Override one lazily resolved ``vllm.envs`` value for the current test.

    The value goes through the environment so ``vllm.envs`` parses it the way
    a launch would; patching the module attribute instead would leave the
    resolved value behind as a permanent attribute after the test, hiding
    later environment changes in the same process.
    """
    disable_envs_cache()
    if isinstance(value, bool):
        value = "1" if value else "0"
    monkeypatch.setenv(name, str(value))


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
        set_ckv_current_cache=lambda *args: None,
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
        # Token-major input scatters heads along dim 1; the SM120 path hands
        # over a head-major copy and scatters along dim 0, then transposes
        # the result back.
        assert dim in (0, 1)
        if dim == 0:
            local_output = local_output.transpose(0, 1)
        torch.testing.assert_close(
            local_output, self.corrected[self.rank_in_group], atol=2e-6, rtol=2e-5
        )
        terms = self.corrected.clone()
        terms[self.rank_in_group] = local_output
        first = self.rank_in_group * _HEADS
        result = terms.sum(0)[:, first : first + _HEADS]
        if dim == 0:
            result = result.transpose(0, 1)
        return result.contiguous()


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


@pytest.mark.parametrize(
    ("cap", "large_backend", "capture", "decodes", "mtp", "full_ckv", "expected"),
    [
        (256, "ag_rs", False, 0, False, False, "ag_rs"),
        (0, "ag_rs", False, 0, False, False, "combine"),
        (-1, "ag_rs", False, 0, False, False, "combine"),
        (256, "a2a", False, 0, False, False, "combine"),
        (256, "ag_rs", True, 0, False, False, "combine"),
        (256, "ag_rs", False, 1, False, False, "ag_rs"),
        (256, "ag_rs", False, 0, True, False, "combine"),
        (256, "ag_rs", False, 0, False, True, None),
    ],
)
@pytest.mark.parametrize("rows", [4, 257])
@pytest.mark.parametrize("spec_decode", [False, True])
def test_prefill_environment_controls_model_exchange(
    monkeypatch,
    cap,
    large_backend,
    capture,
    decodes,
    mtp,
    full_ckv,
    expected,
    rows,
    spec_decode,
):
    """Process-time flags reach model dispatch without changing excluded batches."""
    from vllm.v1.attention.ops import b12x_dcp, dcp

    events: list[str] = []
    module = _module(events, dcp_world_size=4, full_ckv=full_ckv)
    module._is_mtp_layer = mtp
    manager = dcp.MLADCPManager.__new__(dcp.MLADCPManager)
    manager.group = SimpleNamespace(world_size=4)
    manager.use_a2a = True
    manager.is_lse_base_on_e = True
    manager.combine = module.dcp_manager.combine
    manager.query_gather = module.dcp_manager.query_gather

    def ag_rs(output, lse, *, cp_group, is_lse_base_on_e, seq_lens, query_start_loc):
        assert cp_group is manager.group
        assert is_lse_base_on_e
        assert seq_lens.shape == (rows,)
        assert query_start_loc.shape == (rows + 1,)
        events.append("ag_rs")
        return output[:, :_HEADS]

    monkeypatch.setattr(dcp, "cp_lse_ag_out_rs", ag_rs)
    _override_envs(monkeypatch, "VLLM_DCP_A2A_MAX_TOKENS", cap)
    _override_envs(monkeypatch, "VLLM_DCP_A2A_LARGE_BACKEND", large_backend)
    _override_envs(monkeypatch, "VLLM_DCP_PROJECT_BEFORE_MERGE", False)
    _override_envs(monkeypatch, "VLLM_B12X_MLA_DCP_GATHER_IN_WORKSPACE", False)
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8192),
        compilation_config=SimpleNamespace(cudagraph_capture_sizes=[1, 4, 8, 16]),
    )
    module._dcp_prefill_policy = manager.configure_prefill(config)
    module.dcp_manager = manager
    # Changing the environment after construction must not change batch routing.
    _override_envs(monkeypatch, "VLLM_DCP_A2A_MAX_TOKENS", 1)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capture)
    metadata = SimpleNamespace(
        num_actual_tokens=rows,
        is_spec_decode=spec_decode,
        num_prefills=1,
        num_decodes=decodes,
        dcp_combine_seq_lens=torch.ones(rows, dtype=torch.int32),
        dcp_combine_query_start_loc=torch.arange(rows + 1, dtype=torch.int32),
    )
    monkeypatch.setattr(
        attention_module,
        "get_attention_context",
        lambda _: (metadata, None, torch.zeros(1), None),
    )

    def direct_combine(partials, lse, lengths):
        assert lengths is metadata.dcp_combine_seq_lens
        events.append("direct_combine")
        return partials[:, :_HEADS]

    binding = SimpleNamespace(
        accepts=lambda query, lengths: rows <= 16,
        query=manager.query_gather,
        combine=direct_combine,
    )
    monkeypatch.setattr(b12x_dcp, "active_dcp_transport", lambda: binding)
    output = torch.zeros(rows, _HEADS * _V_HEAD)
    module._sparse_indexer_and_attn(
        torch.zeros(rows, 6),
        None,
        None,
        None,
        None,
        None,
        torch.zeros(rows, _HEADS, _LATENT),
        torch.zeros(rows, _HEADS, 3),
        torch.zeros(rows, _HEADS, _ROPE),
        output,
    )
    exchanges = [
        event.split(":")[0]
        for event in events
        if event.startswith(("combine", "ag_rs", "direct_combine"))
    ]
    if spec_decode and not full_ckv:
        expected = "combine"
    if rows <= 16 and not full_ckv:
        expected = "direct_combine"
    assert exchanges == ([] if expected is None else [expected])
    torch.testing.assert_close(output, torch.full_like(output, _LATENT))


def test_projected_prefill_requires_backend_workspace_contract(monkeypatch):
    from vllm.v1.attention.ops import dcp

    _override_envs(monkeypatch, "VLLM_DCP_PROJECT_BEFORE_MERGE", True)
    _override_envs(monkeypatch, "VLLM_B12X_MLA_DCP_GATHER_IN_WORKSPACE", False)
    manager = dcp.MLADCPManager.__new__(dcp.MLADCPManager)
    manager.group = SimpleNamespace(world_size=4)
    manager.use_a2a = True
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8192),
        compilation_config=SimpleNamespace(cudagraph_capture_sizes=[16]),
    )
    with pytest.raises(ValueError, match="backend workspace support"):
        manager.configure_prefill(config)


def test_dcp1_prefill_configuration_ignores_projected_flags(monkeypatch):
    from vllm.v1.attention.ops import dcp
    from vllm.v1.attention.ops.dcp_prefill_policy import DCPPrefillBatch

    _override_envs(monkeypatch, "VLLM_DCP_PROJECT_BEFORE_MERGE", True)
    _override_envs(monkeypatch, "VLLM_B12X_MLA_DCP_GATHER_IN_WORKSPACE", True)
    manager = dcp.MLADCPManager.__new__(dcp.MLADCPManager)
    manager.group = SimpleNamespace(world_size=1)
    manager.use_a2a = True
    manager.is_lse_base_on_e = True
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8192),
        compilation_config=SimpleNamespace(cudagraph_capture_sizes=[16]),
    )
    policy = manager.configure_prefill(config)
    assert policy.select(DCPPrefillBatch(2048, 1, 0)).route == "local"


@pytest.mark.parametrize("borrowed", [False, True])
def test_projected_prefill_uses_reserved_buffers_and_projects_exactly_once(
    monkeypatch, borrowed
):
    """Exercise model→workspace→BMM→RS dispatch with CPU collective substitutes."""
    import sys

    from vllm.v1.attention.ops import dcp, dcp_prefill_workspace
    from vllm.v1.worker.workspace import WorkspaceManager

    rows, capacity = 1025, 1030
    heads = _HEADS * 4
    manager = WorkspaceManager(torch.device("cpu"))
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(
        dcp_prefill_workspace, "current_workspace_manager", lambda: manager
    )
    layer = _module([], dcp_world_size=4, full_ckv=False)
    layer._is_mtp_layer = False
    layer.W_UV = layer.W_UV.bfloat16()
    specs = (
        ((capacity, heads, _LATENT + _ROPE), torch.bfloat16),
        ((capacity * heads * (_LATENT * 2 + 4),), torch.uint8),
    )
    layer.impl.get_dcp_prefill_workspace_specs = lambda: specs
    transport = dcp.MLADCPManager.__new__(dcp.MLADCPManager)
    transport.group = SimpleNamespace(
        world_size=4, rank_in_group=0, device_group=object()
    )
    transport.use_a2a = True
    transport.is_lse_base_on_e = True
    transport.query_gather = lambda *args: pytest.fail(
        "Projected route used allocating query gather"
    )
    transport.combine = lambda *args, **kwargs: pytest.fail(
        "Projected route used latent-output combine"
    )
    for name, value in {
        "VLLM_DCP_PROJECT_BEFORE_MERGE": True,
        "VLLM_B12X_MLA_DCP_GATHER_IN_WORKSPACE": borrowed,
        "VLLM_DCP_PROJECT_BEFORE_MERGE_MIN_PREFILL_TOKENS": 1024,
        "VLLM_DCP_A2A_MAX_TOKENS": 16,
        "VLLM_DCP_A2A_LARGE_BACKEND": "ag_rs",
    }.items():
        _override_envs(monkeypatch, name, value)
    layer._dcp_prefill_policy = transport.configure_prefill(
        SimpleNamespace(
            scheduler_config=SimpleNamespace(max_num_batched_tokens=capacity),
            compilation_config=SimpleNamespace(cudagraph_capture_sizes=[16]),
        ),
        backend=layer.impl,
        latent_dim=_LATENT,
        value_dim=_V_HEAD,
    )
    manager.lock()
    layer.dcp_manager = transport

    def attention(query, *args):
        _, scratch = manager.get_simultaneous(*specs)
        output = (
            scratch[: capacity * heads * _LATENT * 2]
            .view(torch.bfloat16)
            .as_strided((rows, heads, _LATENT), (_LATENT, capacity * _LATENT, 1))
        )
        lse = (
            scratch[capacity * heads * _LATENT * 2 :]
            .view(torch.float32)[: rows * heads]
            .view(rows, heads)
        )
        output.fill_(1)
        output[0] = float("nan")
        lse.zero_()
        return output, lse

    layer.impl.forward_mqa = attention

    def prepare(source, lengths, out):
        out.copy_(torch.where(lengths[:, None] > 0, source, -float("inf")))

    monkeypatch.setitem(
        sys.modules, "b12x.comm.prefill", SimpleNamespace(prepare_prefill_lse=prepare)
    )

    def gather(out, local, *, group):
        for rank in range(4):
            out.view(4, *local.shape)[rank].copy_(local)

    def correct(out, lses, rank, ctx, *, is_lse_base_on_e, lse_output):
        out[0].zero_()
        out.div_(4)
        lse_output.copy_(torch.logsumexp(lses, dim=0))
        return out, lse_output

    def reduce(out, local, *, group):
        out.copy_(local[:_HEADS] * 4)

    monkeypatch.setattr(dcp_prefill_workspace.dist, "all_gather_into_tensor", gather)
    monkeypatch.setattr(dcp_prefill_workspace.dist, "reduce_scatter_tensor", reduce)
    monkeypatch.setattr(dcp, "correct_attn_out", correct)
    metadata = SimpleNamespace(
        num_actual_tokens=rows,
        num_prefills=1,
        num_decodes=0,
        dcp_combine_seq_lens=torch.ones(rows, dtype=torch.int32),
        dcp_combine_query_start_loc=torch.arange(rows + 1, dtype=torch.int32),
    )
    metadata.dcp_combine_seq_lens[0] = 0
    monkeypatch.setattr(
        attention_module,
        "get_attention_context",
        lambda _: (metadata, None, None, None),
    )
    bmm = torch.bmm
    projections = []

    def counted_bmm(*args, **kwargs):
        projections.append(args[0].shape)
        return bmm(*args, **kwargs)

    monkeypatch.setattr(torch, "bmm", counted_bmm)
    output = torch.full((rows + 2, _HEADS * _V_HEAD), 17, dtype=torch.bfloat16)
    layer._sparse_indexer_and_attn(
        torch.zeros(rows, 6),
        None,
        None,
        None,
        None,
        None,
        torch.zeros(rows, _HEADS, _LATENT, dtype=torch.bfloat16),
        torch.zeros(rows, _HEADS, 3, dtype=torch.bfloat16),
        torch.zeros(rows, _HEADS, _ROPE, dtype=torch.bfloat16),
        output,
    )
    assert projections == [torch.Size([heads, rows, _LATENT])]
    expected = torch.full_like(output, _LATENT)
    expected[0].zero_()
    expected[rows:] = 17
    torch.testing.assert_close(output, expected, atol=0, rtol=0)
    assert manager.is_locked()
