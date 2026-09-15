# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Order of the latent query projection against the sparse indexer.

On the bf16 query path the attention module issues the W_UK projection of
the query after the indexer launch, so the selection sort that the B12X
indexer runs on a side stream between the indexer and the attention kernels
has main-stream work to overlap. On the fp8 query path the projection must
precede ``fused_q``, which packs it.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.config import CUDAGraphMode
from vllm.model_executor.layers.attention import mla_attention
from vllm.models.deepseek_v32 import attention as attention_module
from vllm.models.deepseek_v32.attention import DeepseekV32Attention


def _module(events: list[str], *, fp8_query: bool) -> DeepseekV32Attention:
    module = DeepseekV32Attention.__new__(DeepseekV32Attention)
    module.skip_topk = False
    module.use_pcp = False
    module.dcp_manager = None
    module._vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1, cp_kv_cache_interleave_size=1
        )
    )
    module._dense_mha_metadata_layer_name = "dense"
    module.layer_name = "layer"
    module._fp8_query = fp8_query
    module.W_UK_T = torch.ones(2, 3, 4)

    def run_indexer(*args, **kwargs):
        events.append("indexer")

    module.indexer = SimpleNamespace(run_indexer=run_indexer)

    def latent_query(q_nope):
        events.append("latent")
        return torch.zeros(q_nope.shape[0], q_nope.shape[1], 4)

    module._latent_query = latent_query  # type: ignore[method-assign]
    return module


@pytest.mark.parametrize("fp8_query", [False, True])
def test_latent_query_runs_after_the_indexer_on_the_bf16_path(
    monkeypatch: pytest.MonkeyPatch, fp8_query: bool
) -> None:
    events: list[str] = []
    module = _module(events, fp8_query=fp8_query)
    monkeypatch.setattr(
        attention_module,
        "get_attention_context",
        lambda layer_name: (None, None, None, None),
    )
    q_nope = torch.zeros(5, 2, 3)
    output = torch.ones(5, 8)
    ql_nope = module._latent_query(q_nope) if fp8_query else None
    events.clear()
    module._sparse_indexer_and_attn(
        torch.zeros(5, 6),
        torch.zeros(5, 1, 1),
        None,
        torch.zeros(5, 1),
        None,
        None,
        ql_nope,
        q_nope,
        torch.zeros(5, 2, 4),
        output,
    )
    if fp8_query:
        assert events == ["indexer"]
    else:
        assert events == ["indexer", "latent"]
    assert torch.count_nonzero(output) == 0


def _prefill_query_layer(monkeypatch):
    layer = DeepseekV32Attention.__new__(DeepseekV32Attention)
    torch.nn.Module.__init__(layer)
    layer._is_mtp_layer = False
    layer.use_pcp = False
    layer.impl = SimpleNamespace(dcp_world_size=4)
    layer.layer_name = "model.layers.0.self_attn.attn"
    layer.qk_rope_head_dim = 64
    layer.W_UK_T = torch.empty((8, 192, 512), dtype=torch.bfloat16)
    calls = []

    def run(query, weight, output):
        calls.append((query, weight, output))
        output.fill_(7)
        return output

    layer._prefill_query_bmm_module = SimpleNamespace(
        can_implement=lambda **kwargs: 1 <= kwargs["max_m"] <= 8192,
        run=run,
        prewarm=run,
    )
    metadata = SimpleNamespace(
        is_spec_decode=False,
        num_decode_tokens=0,
        num_prefills=1,
        num_actual_tokens=64,
    )
    context = SimpleNamespace(cudagraph_runtime_mode=CUDAGraphMode.NONE)
    monkeypatch.setattr(mla_attention, "get_forward_context", lambda: context)
    monkeypatch.setattr(
        mla_attention,
        "get_attention_context",
        lambda name: (metadata, None, None, None),
    )
    return layer, metadata, context, calls


def test_large_prefill_query_uses_strided_input_and_caller_owned_output(monkeypatch):
    layer, _, _, calls = _prefill_query_layer(monkeypatch)
    packed = torch.empty((64, 8, 256), dtype=torch.bfloat16)
    query = packed[..., :192]

    output = layer._latent_query(query)

    assert len(calls) == 1
    query_arg, weight_arg, output_arg = calls[0]
    assert query_arg.data_ptr() == query.data_ptr()
    assert query_arg.stride() == query.transpose(0, 1).stride()
    assert weight_arg is layer.W_UK_T
    assert output.data_ptr() == output_arg.data_ptr()
    assert output.shape == (64, 8, 512)
    assert torch.all(output == 7)


@pytest.mark.parametrize(
    "excluded",
    [
        "disabled",
        "dcp1",
        "mtp_layer",
        "spec_decode",
        "mixed",
        "no_prefill",
        "padding",
        "small",
        "captured",
        "stream_capture",
        "unsupported",
        "pcp",
        "non_bf16",
    ],
)
def test_prefill_query_bmm_preserves_ineligible_query_paths(monkeypatch, excluded):
    layer, metadata, context, calls = _prefill_query_layer(monkeypatch)
    query = torch.empty((64, 8, 192), dtype=torch.bfloat16)
    if excluded == "disabled":
        layer._prefill_query_bmm_module = None
    elif excluded == "dcp1":
        layer.impl.dcp_world_size = 1
    elif excluded == "mtp_layer":
        layer._is_mtp_layer = True
    elif excluded == "spec_decode":
        metadata.is_spec_decode = True
    elif excluded == "mixed":
        metadata.num_decode_tokens = 1
    elif excluded == "no_prefill":
        metadata.num_prefills = 0
    elif excluded == "padding":
        metadata.num_actual_tokens = 63
    elif excluded == "small":
        query = query[:32]
        metadata.num_actual_tokens = 32
    elif excluded == "captured":
        context.cudagraph_runtime_mode = CUDAGraphMode.PIECEWISE
    elif excluded == "stream_capture":
        monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    elif excluded == "unsupported":
        layer._prefill_query_bmm_module.can_implement = lambda **kwargs: False
    elif excluded == "pcp":
        layer.use_pcp = True
    elif excluded == "non_bf16":
        query = query.float()
    fallback_calls = []

    def fallback(query_arg, weight):
        fallback_calls.append((query_arg, weight))
        return torch.zeros((8, query_arg.shape[1], 512), dtype=query_arg.dtype)

    monkeypatch.setattr(torch, "bmm", fallback)
    output = layer._latent_query(query)
    assert not calls
    assert len(fallback_calls) == 1
    assert output.shape == (query.shape[0], 8, 512)


@pytest.mark.parametrize("mtp_layer", [False, True])
def test_prefill_query_warmup_covers_supported_target_rows(monkeypatch, mtp_layer):
    layer, _, _, calls = _prefill_query_layer(monkeypatch)
    layer._is_mtp_layer = mtp_layer
    monkeypatch.setattr(
        mla_attention, "can_implement_bf16_mla_query", lambda **kwargs: False
    )
    assert layer.prepare_dcp_prefill((1, 32, 33, 64, 8193)) is (not mtp_layer)
    assert len(calls) == (0 if mtp_layer else 1)
    if not mtp_layer:
        query, weight, output = calls[0]
        assert query.shape == (8, 64, 192)
        assert query.stride() == (256, 2048, 1)
        assert weight is layer.W_UK_T
        assert output.shape == (8, 64, 512)
    layer._prefill_query_bmm_module = None
    assert not layer.prepare_dcp_prefill((1, 32, 33, 64, 8193))


def test_prefill_collective_warmup_prepares_backend_before_projected_exchange(
    monkeypatch,
):
    layer, _, _, _ = _prefill_query_layer(monkeypatch)
    layer._prefill_query_bmm_module = None
    layer.W_UV = torch.empty((8, 512, 256), dtype=torch.bfloat16)
    monkeypatch.setattr(
        mla_attention, "can_implement_bf16_mla_query", lambda **kwargs: False
    )
    events = []
    specs = (((64, 32, 576), torch.bfloat16), ((1024,), torch.uint8))
    layer.impl.prepare_profile_collectives = lambda: events.append("backend")
    layer.impl.get_dcp_prefill_workspace_specs = lambda: specs

    def prewarm(weight, *, backend_specs):
        assert weight is layer.W_UV
        assert backend_specs is specs
        events.append("collectives")

    layer.dcp_manager = SimpleNamespace(
        prefill_warmup_key=((0, 1, 2, 3), "borrowed"),
        prewarm_prefill=prewarm,
    )
    assert layer.prepare_dcp_prefill((64,))
    assert events == ["backend", "collectives"]
    layer.dcp_manager.prefill_warmup_key = None
    events.clear()
    layer.prepare_dcp_prefill((64,))
    assert events == ["backend"]
