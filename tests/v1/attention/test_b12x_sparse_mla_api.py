# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Behavior checks for the B12x sparse MLA adapters."""

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm.config import AttentionConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.attention.mla_attention import (
    MLAAttention,
    _canonicalize_sparse_mla_kv_cache_dtype,
    _maybe_view_mla_cache_as_fp8,
    _uses_packed_sparse_mla_workspace,
)
from vllm.model_executor.layers.attention.sparse_mla_attention import (
    SparseMLACommonMetadataBuilder,
)
from vllm.models.deepseek_v4.nvidia import b12x as b12x_mla
from vllm.models.deepseek_v4.nvidia import b12x_indexer
from vllm.models.deepseek_v32.nvidia.b12x import (
    B12xDSAIndexer,
    DeepseekV32B12xAttention,
    DeepseekV32B12xIndexerAttention,
    _get_sparse_mla_backend,
)
from vllm.models.deepseek_v32.nvidia.model import _get_attention_cls
from vllm.platforms.interface import DeviceCapability, Platform
from vllm.v1.attention.backends.b12x import B12xPagedAttentionBackend
from vllm.v1.attention.backends.mla import b12x_indexer as generic_b12x_indexer
from vllm.v1.attention.backends.mla import b12x_mla_sparse
from vllm.v1.attention.backends.mla.b12x_indexer import B12xIndexerBackend
from vllm.v1.attention.backends.mla.b12x_mla_sparse import (
    B12xGLM5NextMLASparseBackend,
    B12xGLM5NextMLASparseMetadataBuilder,
    B12xGLMDSAMLASparseBackend,
    B12xMLASparseBackend,
    B12xMLASparseImpl,
    B12xMLASparseMetadata,
    B12xMLASparseMetadataBuilder,
    _ckv_rank_token_alignment,
    _global_causal_lens_for_ckv_gather,
    _is_native_ckv_source_layout,
    _is_speculative_decode_batch,
    _max_speculative_decode_query_len,
    _round_up_ckv_rank_tokens,
    _selected_index_block_stride_rows,
    _use_b12x_full_ckv_gather,
)
from vllm.v1.attention.backends.mla.sparse_utils import _remap_tiling
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.kv_cache_interface import MLAAttentionSpec
from vllm.v1.kv_cache_layout import KVCacheLayout
from vllm.v1.worker.utils import select_common_block_size


class _Workspace:
    def get_simultaneous(self, *shapes_and_dtypes):
        return [torch.empty(shape, dtype=dtype) for shape, dtype in shapes_and_dtypes]


def test_b12x_selector_routes_supported_attention_families(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_USE_B12X_SPARSE_INDEXER", "0")
    assert AttentionConfig(backend="b12x").backend == AttentionBackendEnum.B12X
    assert AttentionBackendEnum.B12X.get_class() is B12xPagedAttentionBackend
    assert B12xMLASparseBackend.get_name() == "B12X"
    assert b12x_mla.DeepseekV4B12xSparseMLABackend.get_name() == "B12X"
    assert not B12xIndexerBackend.supports_device_cpu_query_lens_mismatch()
    assert not B12xMLASparseBackend.supports_device_cpu_query_lens_mismatch()

    config = SimpleNamespace(
        attention_config=SimpleNamespace(backend=AttentionBackendEnum.B12X)
    )
    assert _get_attention_cls(config) is DeepseekV32B12xAttention
    assert DeepseekV32B12xAttention.indexer_cls is B12xDSAIndexer

    config.model_config = SimpleNamespace(
        hf_text_config=SimpleNamespace(model_type="glm_moe_dsa")
    )
    assert _get_sparse_mla_backend(config) is B12xGLMDSAMLASparseBackend


def test_b12x_indexer_selector_preserves_sparse_mla_backend(monkeypatch) -> None:
    config = SimpleNamespace(
        attention_config=SimpleNamespace(
            backend=AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM120
        )
    )

    monkeypatch.setenv("VLLM_USE_B12X_SPARSE_INDEXER", "0")
    assert _get_attention_cls(config) is not DeepseekV32B12xIndexerAttention

    monkeypatch.setenv("VLLM_USE_B12X_SPARSE_INDEXER", "1")
    attention_cls = _get_attention_cls(config)
    assert attention_cls is DeepseekV32B12xIndexerAttention
    assert attention_cls.indexer_cls is B12xDSAIndexer
    assert "__init__" not in attention_cls.__dict__
    assert (
        config.attention_config.backend
        == AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM120
    )


def test_b12x_sparse_mla_accepts_glm_dsa_contract(monkeypatch) -> None:
    monkeypatch.setattr(b12x_mla_sparse, "get_b12x_sparse_mla", lambda: object())
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(
                model_type="glm_moe_dsa",
                index_topk=2048,
                kv_lora_rank=512,
                qk_rope_head_dim=64,
                qk_nope_head_dim=192,
                v_head_dim=256,
            )
        )
    )

    with set_current_vllm_config(config):
        invalid_reasons = B12xMLASparseBackend.validate_configuration(
            head_size=576,
            dtype=torch.bfloat16,
            kv_cache_dtype="fp8",
            block_size=64,
            use_mla=True,
            has_sink=False,
            use_sparse=True,
            use_mm_prefix=False,
            use_per_head_quant_scales=False,
            device_capability=DeviceCapability(12, 0),
            attn_type="decoder",
        )

    assert invalid_reasons == []
    assert (
        _canonicalize_sparse_mla_kv_cache_dtype(B12xMLASparseBackend, "auto")
        == "fp8_ds_mla"
    )

    with set_current_vllm_config(config):
        invalid_reasons = B12xMLASparseBackend.validate_configuration(
            head_size=576,
            dtype=torch.bfloat16,
            kv_cache_dtype="nvfp4_ds_mla",
            block_size=64,
            use_mla=True,
            has_sink=False,
            use_sparse=True,
            use_mm_prefix=False,
            use_per_head_quant_scales=False,
            device_capability=DeviceCapability(12, 0),
            attn_type="decoder",
        )

    assert invalid_reasons == []


def test_b12x_glm_dsa_nvfp4_cache_spec(monkeypatch) -> None:
    monkeypatch.setattr(b12x_mla_sparse, "get_b12x_sparse_mla", lambda: object())
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(model_type="glm_moe_dsa")
        )
    )
    probe = MLAAttentionSpec(
        block_size=64,
        num_kv_heads=1,
        head_size=576,
        dtype=torch.uint8,
        cache_dtype_str="nvfp4_ds_mla",
    )

    with set_current_vllm_config(config):
        packed = B12xMLASparseBackend.customize_spec(probe)

    assert packed.state_content_bytes == 368
    assert packed.page_size_bytes == 64 * 368
    assert packed.model_version == "glm_moe_dsa"
    assert B12xMLASparseBackend.customize_spec(packed) == packed

    packed_without_config = B12xGLMDSAMLASparseBackend.customize_spec(probe)
    assert packed_without_config == packed


def test_b12x_nvfp4_rejects_non_glm_dsa_architecture(monkeypatch) -> None:
    monkeypatch.setattr(b12x_mla_sparse, "get_b12x_sparse_mla", lambda: object())
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(
                model_type="deepseek_v32",
                index_topk=2048,
                kv_lora_rank=512,
                qk_rope_head_dim=64,
            )
        )
    )

    with set_current_vllm_config(config):
        invalid_reasons = B12xMLASparseBackend.validate_configuration(
            head_size=576,
            dtype=torch.bfloat16,
            kv_cache_dtype="nvfp4_ds_mla",
            block_size=64,
            use_mla=True,
            has_sink=False,
            use_sparse=True,
            use_mm_prefix=False,
            use_per_head_quant_scales=False,
            device_capability=DeviceCapability(12, 0),
            attn_type="decoder",
        )

    assert invalid_reasons == [
        ("B12X nvfp4_ds_mla requires GLM5Next or the GLM-5.2/5.3 DSA architecture")
    ]


def test_b12x_dsa_requires_layer_compact_cache_layout() -> None:
    assert B12xMLASparseBackend.supported_kv_cache_layouts() == (KVCacheLayout.LBNHC,)


def test_b12x_glm5_next_requires_block_outermost_cache_layout() -> None:
    assert B12xGLM5NextMLASparseBackend.supported_kv_cache_layouts() == (
        KVCacheLayout.BLHNC,
    )


def _glm5_next_config(
    *,
    dcp_size: int = 1,
    cp_interleave: int = 1,
    speculative: bool = False,
    prefix_caching: bool = False,
    **overrides: int,
) -> SimpleNamespace:
    recipe = dict(
        model_type="glm5_next_text",
        kv_lora_rank=512,
        qk_nope_head_dim=256,
        qk_rope_head_dim=0,
        v_head_dim=256,
        index_n_heads=32,
        index_head_dim=128,
        index_topk=2048,
        index_kpool=4,
    )
    recipe.update(overrides)
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(**recipe)),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=dcp_size,
            cp_kv_cache_interleave_size=cp_interleave,
        ),
        speculative_config=object() if speculative else None,
        cache_config=SimpleNamespace(enable_prefix_caching=prefix_caching),
    )


def test_b12x_glm5_next_cache_spec_and_layout(monkeypatch) -> None:
    monkeypatch.setattr(b12x_mla_sparse, "get_b12x_sparse_mla", lambda: object())
    config = _glm5_next_config()
    probe = MLAAttentionSpec(
        block_size=64,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.uint8,
        cache_dtype_str="fp8_ds_mla",
        state_content_bytes=656,
    )
    unidentified = B12xMLASparseBackend.customize_spec(probe)
    packed_by_glm_backend = B12xGLM5NextMLASparseBackend.customize_spec(probe)
    with set_current_vllm_config(config):
        invalid_reasons = B12xMLASparseBackend.validate_configuration(
            head_size=512,
            dtype=torch.bfloat16,
            kv_cache_dtype="fp8",
            block_size=64,
            use_mla=True,
            has_sink=False,
            use_sparse=True,
            use_mm_prefix=False,
            use_per_head_quant_scales=False,
            device_capability=DeviceCapability(12, 0),
            attn_type="decoder",
        )
        packed = B12xMLASparseBackend.customize_spec(probe)
        layouts = B12xGLM5NextMLASparseBackend.supported_kv_cache_layouts()
    packed_without_config_context = B12xMLASparseBackend.customize_spec(packed)

    assert invalid_reasons == []
    assert unidentified == probe
    assert packed_by_glm_backend.state_content_bytes == 528
    assert packed_by_glm_backend.page_size_padded is None
    assert packed_by_glm_backend.page_tail_bytes_per_token == 132 // 4
    assert packed_by_glm_backend.page_size_bytes == 64 * (528 + 132 // 4)
    assert packed_by_glm_backend.model_version == "glm5_next"
    assert packed.state_content_bytes == 528
    assert packed.page_size_padded is None
    assert packed.page_tail_bytes_per_token == 132 // 4
    assert packed.page_size_bytes == 64 * (528 + 132 // 4)
    assert packed.model_version == "glm5_next"
    assert packed_without_config_context == packed
    assert layouts == (KVCacheLayout.BLHNC,)
    assert "nvfp4_ds_mla" in B12xMLASparseBackend.supported_kv_cache_dtypes


def test_b12x_glm5_next_nvfp4_cache_spec() -> None:
    probe = MLAAttentionSpec(
        block_size=64,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.uint8,
        cache_dtype_str="nvfp4_ds_mla",
        state_content_bytes=656,
    )

    with set_current_vllm_config(_glm5_next_config()):
        packed = B12xMLASparseBackend.customize_spec(probe)

    assert packed.state_content_bytes == 304
    assert packed.page_tail_bytes_per_token == 33
    assert packed.page_size_padded == 3 * 64 * 132
    assert packed.page_size_bytes == 3 * 64 * 132 + 64 * 33
    assert packed.model_version == "glm5_next"


def test_b12x_glm5_next_binds_nvfp4_record_width(monkeypatch) -> None:
    impl = object.__new__(B12xMLASparseImpl)
    impl._is_glm_next = True
    impl._cache_record_bytes = 304
    impl._ckv_gather_enabled = False
    planned: list[int] = []
    monkeypatch.setattr(impl, "_set_kernel_page_size", planned.append)

    impl.bind_kv_cache(torch.empty((2, 64, 304), dtype=torch.uint8))

    assert planned == [64]
    with pytest.raises(ValueError, match="page_size, 304"):
        impl.bind_kv_cache(torch.empty((2, 64, 528), dtype=torch.uint8))


def test_b12x_glm_dsa_binds_nvfp4_fp8_rope_record() -> None:
    impl = object.__new__(B12xMLASparseImpl)
    impl._is_glm_next = False
    impl._uses_glm_dsa_nvfp4_cache = True

    impl.bind_kv_cache(torch.empty((2, 64, 368), dtype=torch.uint8))

    with pytest.raises(ValueError, match="page_size, 368"):
        impl.bind_kv_cache(torch.empty((2, 64, 432), dtype=torch.uint8))


@pytest.mark.parametrize(
    ("model_type", "kv_cache_dtype", "record_bytes"),
    [
        ("glm_moe_dsa", "fp8_ds_mla", 656),
        ("deepseek_v32", "fp8_ds_mla", 656),
        ("glm5_next_text", "fp8_ds_mla", 528),
        ("glm_moe_dsa", "nvfp4_ds_mla", 368),
        ("glm5_next_text", "nvfp4_ds_mla", 304),
    ],
)
@pytest.mark.parametrize("dcp_size", [1, 4])
def test_b12x_sparse_mla_constructor_uses_planned_cache_width(
    monkeypatch, model_type: str, kv_cache_dtype: str, record_bytes: int, dcp_size: int
) -> None:
    sparse_mla = pytest.importorskip("b12x.attention.sparse_mla")
    is_glm_next = model_type == "glm5_next_text"
    config = _glm5_next_config(dcp_size=dcp_size, cp_interleave=4)
    config.parallel_config.enable_dbo = False
    monkeypatch.setenv("VLLM_B12X_MLA_CKV_GATHER", "1")
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_dcp_group",
        lambda: SimpleNamespace(world_size=dcp_size, rank_in_group=0),
    )
    hf_config = config.model_config.hf_text_config
    hf_config.model_type = model_type
    hf_config.qk_nope_head_dim = 256 if is_glm_next else 192
    hf_config.qk_rope_head_dim = 0 if is_glm_next else 64
    config.cache_config.block_size = 64
    config.scheduler_config = SimpleNamespace(max_num_batched_tokens=16, max_num_seqs=2)
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)
    monkeypatch.setattr(
        "vllm.model_executor.layers.attention.sparse_mla_attention."
        "get_tensor_model_parallel_world_size",
        lambda: 8,
    )
    monkeypatch.setattr(b12x_mla_sparse, "get_b12x_sparse_mla", lambda: sparse_mla)
    monkeypatch.setattr(
        b12x_mla_sparse, "is_workspace_manager_initialized", lambda: False
    )
    monkeypatch.setattr(sparse_mla, "is_supported", lambda: True)
    # Resolve the real cache contract without compiling GPU kernels.
    monkeypatch.setattr(
        sparse_mla,
        "plan",
        lambda caps: SimpleNamespace(caps=caps, layout=SimpleNamespace(nbytes=256)),
    )
    topk_width = hf_config.index_topk + (
        hf_config.index_kpool - 1 if is_glm_next else 0
    )
    with set_current_vllm_config(config):
        impl = B12xMLASparseImpl(
            num_heads=8,
            head_size=512 + hf_config.qk_rope_head_dim,
            scale=256**-0.5,
            num_kv_heads=1,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype=kv_cache_dtype,
            logits_soft_cap=None,
            attn_type="decoder",
            kv_sharing_target_layer_name=None,
            q_lora_rank=2048,
            kv_lora_rank=512,
            qk_nope_head_dim=hf_config.qk_nope_head_dim,
            qk_rope_head_dim=hf_config.qk_rope_head_dim,
            qk_head_dim=256,
            v_head_dim=256,
            kv_b_proj=None,
            topk_indices_buffer=torch.empty((16, topk_width), dtype=torch.int32),
        )

    assert impl._cache_record_bytes == record_bytes
    assert impl._decode_plan.caps.cache_record_bytes == record_bytes
    assert impl._extend_plan.caps.cache_record_bytes == record_bytes
    expected_gather = dcp_size > 1 and model_type != "deepseek_v32"
    assert impl._ckv_gather_enabled is expected_gather
    if expected_gather:
        assert impl._ckv_extend_plan.caps.num_q_heads == 8
        assert impl._ckv_extend_plan.caps.cache_record_bytes == record_bytes
        assert impl._extend_plan.caps.num_q_heads == 8 * dcp_size
    else:
        assert impl._ckv_extend_plan is None


def test_b12x_nvfp4_run_options_match_each_glm_record_abi() -> None:
    assert b12x_mla_sparse._nvfp4_run_options(is_glm_next=False) == {
        "scale_format": 2,
        "fp8_rope": True,
    }
    assert b12x_mla_sparse._nvfp4_run_options(is_glm_next=True) == {
        "scale_format": 2,
        "fp8_rope": False,
        "latent_scale_per_token": True,
    }


def test_packed_nvfp4_mla_dtype_bypasses_generic_layout_guard() -> None:
    config = SimpleNamespace(
        model_config=SimpleNamespace(use_mla=True),
        cache_config=SimpleNamespace(cache_dtype="nvfp4_ds_mla"),
    )

    assert VllmConfig.validate_nvfp4_kv_cache_with_mla(config) is config

    config.cache_config.cache_dtype = "nvfp4"
    with pytest.raises(ValueError, match="not supported with MLA"):
        VllmConfig.validate_nvfp4_kv_cache_with_mla(config)


@pytest.mark.parametrize("cache_dtype", ["fp8_ds_mla", "nvfp4_ds_mla"])
def test_packed_mla_cache_keeps_uint8_forward_view(cache_dtype: str) -> None:
    cache = torch.empty((2, 64, 304), dtype=torch.uint8)

    forwarded = _maybe_view_mla_cache_as_fp8(cache, cache_dtype)

    assert forwarded is cache
    assert forwarded.dtype == torch.uint8


def test_plain_fp8_mla_cache_uses_native_fp8_forward_view(monkeypatch) -> None:
    cache = torch.empty((2, 64, 512), dtype=torch.uint8)
    monkeypatch.setattr(
        "vllm.model_executor.layers.attention.mla_attention.current_platform.fp8_dtype",
        lambda: torch.float8_e4m3fn,
    )

    forwarded = _maybe_view_mla_cache_as_fp8(cache, "fp8")

    assert forwarded.data_ptr() == cache.data_ptr()
    assert forwarded.dtype == torch.float8_e4m3fn


@pytest.mark.parametrize(
    ("resolved_cache_dtype", "expected"),
    [
        ("fp8_ds_mla", True),
        ("nvfp4_ds_mla", True),
        ("auto", False),
        (None, False),
    ],
)
def test_packed_workspace_uses_resolved_layer_cache_spec(
    resolved_cache_dtype: str | None,
    expected: bool,
) -> None:
    spec = SimpleNamespace(cache_dtype_str=resolved_cache_dtype)

    assert _uses_packed_sparse_mla_workspace(spec) is expected


def test_b12x_glm5_next_keeps_hybrid_manager_page_unsplit() -> None:
    supported = B12xGLM5NextMLASparseBackend.get_supported_kernel_block_sizes()

    assert len(supported) == 1
    assert supported[0].base == 64
    assert select_common_block_size(2304, [B12xGLM5NextMLASparseBackend]) == 2304
    assert B12xGLM5NextMLASparseBackend.supported_kv_cache_layouts() == (
        KVCacheLayout.BLHNC,
    )


def test_glm5_next_split_cache_auto_aligns_to_dcp_retention(monkeypatch) -> None:
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            architecture="Glm5NextForConditionalGeneration",
        ),
        parallel_config=SimpleNamespace(decode_context_parallel_size=4),
        cache_config=SimpleNamespace(
            block_size=256,
            mamba_block_size=None,
            mamba_cache_mode="align",
            mamba_page_size_padded=1234,
            prefix_cache_retention_interval=4096,
        ),
    )
    monkeypatch.setenv("VLLM_GLM53_SPLIT_TARGET_BLOCK_SIZE", "auto")
    monkeypatch.setenv("VLLM_GLM53_SPLIT_MAMBA_BLOCK_SIZE", "auto")

    Platform._align_hybrid_block_size(config, B12xGLM5NextMLASparseBackend)

    assert config.cache_config.block_size == 1024
    assert config.cache_config.mamba_block_size == 1024
    assert config.cache_config.mamba_page_size_padded is None


@pytest.mark.parametrize(
    (
        "dcp",
        "retention_interval",
        "scheduled_tokens",
        "batched_tokens",
        "expected_block_size",
    ),
    [
        (1, None, None, 4096, 4096),
        (2, None, None, 4096, 2048),
        (4, None, None, 4096, 1024),
        (8, None, None, 4096, 512),
        (4, 0, None, 4096, 1024),
        (4, 0, 4096, 4352, 1024),
    ],
)
def test_glm5_next_split_cache_auto_falls_back_to_scheduler_budget(
    monkeypatch,
    dcp: int,
    retention_interval: int | None,
    scheduled_tokens: int | None,
    batched_tokens: int,
    expected_block_size: int,
) -> None:
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            architecture="Glm5NextForConditionalGeneration",
        ),
        parallel_config=SimpleNamespace(decode_context_parallel_size=dcp),
        scheduler_config=SimpleNamespace(
            max_num_scheduled_tokens=scheduled_tokens,
            max_num_batched_tokens=batched_tokens,
        ),
        cache_config=SimpleNamespace(
            block_size=256,
            mamba_block_size=None,
            mamba_cache_mode="align",
            mamba_page_size_padded=1234,
            prefix_cache_retention_interval=retention_interval,
        ),
    )
    monkeypatch.setenv("VLLM_GLM53_SPLIT_TARGET_BLOCK_SIZE", "auto")
    monkeypatch.setenv("VLLM_GLM53_SPLIT_MAMBA_BLOCK_SIZE", "auto")

    Platform._align_hybrid_block_size(config, B12xGLM5NextMLASparseBackend)

    assert config.cache_config.block_size == expected_block_size
    assert config.cache_config.mamba_block_size == expected_block_size
    assert config.cache_config.mamba_page_size_padded is None


def test_glm5_next_split_cache_auto_requires_dcp_aligned_retention(
    monkeypatch,
) -> None:
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            architecture="Glm5NextForConditionalGeneration",
        ),
        parallel_config=SimpleNamespace(decode_context_parallel_size=4),
        cache_config=SimpleNamespace(
            mamba_cache_mode="align",
            prefix_cache_retention_interval=4097,
        ),
    )
    monkeypatch.setenv("VLLM_GLM53_SPLIT_TARGET_BLOCK_SIZE", "auto")

    with pytest.raises(ValueError, match="divisible by decode_context_parallel_size"):
        Platform._align_hybrid_block_size(config, B12xGLM5NextMLASparseBackend)


def test_b12x_glm5_next_nvfp4_aligns_hybrid_page_to_packed_record(
    monkeypatch,
) -> None:
    mamba_page_size = 1_085_440
    config = _glm5_next_config(dcp_size=4, cp_interleave=4)
    config.model_config.is_hybrid = True
    config.model_config.use_mla = True
    config.model_config.architecture = "Glm5NextForConditionalGeneration"
    config.model_config.dtype = torch.bfloat16
    config.model_config.get_num_kv_heads = lambda parallel_config: 1
    config.model_config.get_head_size = lambda: 512
    config.cache_config.cache_dtype = "nvfp4_ds_mla"
    config.cache_config.block_size = 256
    config.cache_config.mamba_block_size = None
    config.cache_config.user_specified_mamba_block_size = False
    config.cache_config.mamba_cache_mode = "align"
    config.cache_config.mamba_page_size_padded = None

    model_cls = SimpleNamespace(
        get_mamba_state_shape_from_config=lambda vllm_config: ((mamba_page_size,),),
        get_mamba_state_dtype_from_config=lambda vllm_config: (torch.uint8,),
    )
    monkeypatch.setattr(
        "vllm.model_executor.models.ModelRegistry.resolve_model_cls",
        lambda *args, **kwargs: (model_cls, None),
    )

    Platform._align_hybrid_block_size(config, B12xGLM5NextMLASparseBackend)

    materialized_probe = MLAAttentionSpec(
        block_size=config.cache_config.block_size,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.uint8,
        cache_dtype_str="nvfp4_ds_mla",
        state_content_bytes=656,
    )
    with set_current_vllm_config(config):
        materialized = B12xGLM5NextMLASparseBackend.customize_spec(materialized_probe)

    assert config.cache_config.block_size == 3328
    assert config.cache_config.mamba_block_size == 3328
    assert materialized.page_size_bytes == 1_123_584
    assert config.cache_config.mamba_page_size_padded == materialized.page_size_bytes


def test_b12x_glm5_next_rejects_unaligned_dcp(monkeypatch) -> None:
    monkeypatch.setattr(b12x_mla_sparse, "get_b12x_sparse_mla", lambda: object())
    with set_current_vllm_config(_glm5_next_config(dcp_size=2)):
        invalid_reasons = B12xMLASparseBackend.validate_configuration(
            head_size=512,
            dtype=torch.bfloat16,
            kv_cache_dtype="fp8",
            block_size=64,
            use_mla=True,
            has_sink=False,
            use_sparse=True,
            use_mm_prefix=False,
            use_per_head_quant_scales=False,
            device_capability=DeviceCapability(12, 0),
            attn_type="decoder",
        )

    assert invalid_reasons == [
        "B12X GLM5Next C4 DCP requires cp_kv_cache_interleave_size divisible by 4"
    ]


def test_b12x_glm5_next_accepts_pool_aligned_dcp_without_speculation(
    monkeypatch,
) -> None:
    monkeypatch.setattr(b12x_mla_sparse, "get_b12x_sparse_mla", lambda: object())
    with set_current_vllm_config(_glm5_next_config(dcp_size=4, cp_interleave=4)):
        invalid_reasons = B12xMLASparseBackend.validate_configuration(
            head_size=512,
            dtype=torch.bfloat16,
            kv_cache_dtype="fp8",
            block_size=64,
            use_mla=True,
            has_sink=False,
            use_sparse=True,
            use_mm_prefix=False,
            use_per_head_quant_scales=False,
            device_capability=DeviceCapability(12, 0),
            attn_type="decoder",
        )

    assert invalid_reasons == []


def test_b12x_glm5_next_accepts_dcp_with_speculation(monkeypatch) -> None:
    monkeypatch.setattr(b12x_mla_sparse, "get_b12x_sparse_mla", lambda: object())
    with set_current_vllm_config(
        _glm5_next_config(dcp_size=4, cp_interleave=4, speculative=True)
    ):
        invalid_reasons = B12xMLASparseBackend.validate_configuration(
            head_size=512,
            dtype=torch.bfloat16,
            kv_cache_dtype="fp8",
            block_size=64,
            use_mla=True,
            has_sink=False,
            use_sparse=True,
            use_mm_prefix=False,
            use_per_head_quant_scales=False,
            device_capability=DeviceCapability(12, 0),
            attn_type="decoder",
        )

    assert invalid_reasons == []


@pytest.mark.parametrize(
    ("max_query_len", "num_decode_tokens", "num_tokens", "expected"),
    [
        (1, 0, 32, False),
        (6, 192, 192, False),
        (6, 0, 192, True),
        (128, 1, 8192, False),
        (128, 0, 16, False),
        (128, 0, 17, True),
        (128, 0, 8192, True),
        (128, 0, 524288, True),
        (128, 0, 524289, False),
        (128, 0, 600000, False),
    ],
)
def test_b12x_full_ckv_gather_excludes_decode_and_mtp_batches(
    max_query_len: int,
    num_decode_tokens: int,
    num_tokens: int,
    expected: bool,
) -> None:
    assert (
        _use_b12x_full_ckv_gather(
            enabled=True,
            is_glm_next=True,
            dcp_world_size=4,
            max_query_len=max_query_len,
            num_tokens=num_tokens,
            num_decode_tokens=num_decode_tokens,
            min_tokens=16,
            max_tokens=524288,
        )
        is expected
    )


@pytest.mark.parametrize(
    ("seq_lens", "query_starts", "request_ids", "expected"),
    [
        ([5, 12], [0, 2, 5], [0, 0, 1, 1, 1], [4, 5, 10, 11, 12]),
        ([3, 1], [0, 3, 4], [0, 0, 0, 1], [1, 2, 3, 1]),
        ([17, 8, 33], [0, 2, 3, 6], [0, 0, 1, 2, 2, 2], [16, 17, 8, 31, 32, 33]),
        ([8, 17], [0, 1, 3], [0, 1, 1], [8, 16, 17]),
        ([0], [0, 0], [], []),
    ],
)
def test_b12x_full_ckv_gather_uses_global_causal_lengths(
    seq_lens: list[int],
    query_starts: list[int],
    request_ids: list[int],
    expected: list[int],
) -> None:
    global_seq_lens = torch.tensor(seq_lens, dtype=torch.int32)
    query_start_loc = torch.tensor(query_starts + [-1], dtype=torch.int32)
    req_id_per_token = torch.tensor(request_ids + [-1, -1], dtype=torch.int32)

    actual = _global_causal_lens_for_ckv_gather(
        global_seq_lens,
        query_start_loc,
        req_id_per_token,
        num_actual_tokens=len(expected),
    )

    assert actual.dtype == torch.int32
    assert actual.tolist() == expected


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_full_ckv_mapping_preserves_input_order_and_invalid_tail(
    monkeypatch: pytest.MonkeyPatch, device: str
) -> None:
    from b12x.attention._shared.mla import dcp_ckv_mapping

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required for the package kernel integration")
    req_ids = torch.tensor([0, 1, 0], dtype=torch.int32, device=device)
    indices = torch.full((3, 2048), -1, dtype=torch.int32, device=device)
    indices[0, [0, 127, 128, 2047]] = torch.tensor(
        [8, 1, 4, 0], dtype=torch.int32, device=device
    )
    indices[1, [0, 128, 2047]] = torch.tensor(
        [7, 0, 4], dtype=torch.int32, device=device
    )
    starts = torch.tensor(
        [[0, 3], [0, 2], [0, 2], [0, 2]], dtype=torch.int32, device=device
    )
    lengths = torch.tensor(
        [[3, 2], [2, 2], [2, 2], [2, 2]], dtype=torch.int32, device=device
    )
    out = torch.full_like(indices, 99)
    counts = torch.full_like(req_ids, 99)
    expected = torch.full_like(indices, -1)
    expected[0, :4] = torch.tensor([2, 10, 1, 0], dtype=torch.int32, device=device)
    expected[1, :3] = torch.tensor([33, 3, 4], dtype=torch.int32, device=device)
    expected_counts = torch.tensor([4, 3, 0], dtype=torch.int32, device=device)
    originals = [tensor.clone() for tensor in (req_ids, indices, starts, lengths)]
    calls = []
    if device == "cpu":

        def package_dispatch(*args, **kwargs):
            assert all(
                a is b
                for a, b in zip(args, (req_ids, indices, starts, lengths, out, counts))
            )
            assert kwargs == dict(
                dcp_size=4, cp_kv_cache_interleave_size=1, padded_rank_tokens=10
            )
            calls.append(True)
            out.copy_(expected)
            counts.copy_(expected_counts)

        monkeypatch.setattr(
            dcp_ckv_mapping, "map_global_topk_to_gathered_ckv", package_dispatch
        )
    for _ in range(3):
        b12x_mla_sparse._map_global_topk_to_gathered_ckv(
            req_ids,
            indices,
            starts,
            lengths,
            out,
            counts,
            dcp_size=4,
            cp_kv_cache_interleave_size=1,
            padded_rank_tokens=10,
        )
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
        torch.testing.assert_close(counts, expected_counts, rtol=0, atol=0)
    for actual, original in zip((req_ids, indices, starts, lengths), originals):
        torch.testing.assert_close(actual, original, rtol=0, atol=0)
    if device == "cpu":
        assert len(calls) == 3


@pytest.mark.parametrize(
    "invalid", ["output_shape", "rank_shape", "rank_count", "dtype"]
)
def test_full_ckv_mapping_rejects_incompatible_metadata(invalid: str) -> None:
    req_ids = torch.zeros(1, dtype=torch.int32)
    indices = torch.zeros((1, 2048), dtype=torch.int32)
    starts = torch.zeros((4, 1), dtype=torch.int32)
    lengths = torch.ones_like(starts)
    out = torch.empty_like(indices)
    counts = torch.empty_like(req_ids)
    if invalid == "output_shape":
        out = out[:, :128]
    elif invalid == "rank_shape":
        lengths = lengths[:3]
    elif invalid == "rank_count":
        starts, lengths = starts[:3], lengths[:3]
    else:
        counts = counts.to(torch.int64)
    with pytest.raises(TypeError if invalid == "dtype" else ValueError):
        b12x_mla_sparse._map_global_topk_to_gathered_ckv(
            req_ids,
            indices,
            starts,
            lengths,
            out,
            counts,
            dcp_size=4,
            cp_kv_cache_interleave_size=1,
            padded_rank_tokens=10,
        )


def test_b12x_full_ckv_gather_capture_fallback_ignores_runtime_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    impl = object.__new__(B12xMLASparseImpl)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

    assert not impl.uses_full_ckv_dcp(SimpleNamespace(), 128)


@pytest.mark.parametrize("depth", [0, 1])
@pytest.mark.parametrize("profile_runner", [None, "v1", "v2"])
def test_full_ckv_collective_warmup_covers_reserved_slots_lanes_and_streams(
    monkeypatch, depth, profile_runner
) -> None:
    from vllm.v1.attention.backends.mla.ckv_prefetch import (
        CKVPrefetchPlan,
        CKVPrefetchRegistry,
        CKVWorkspacePool,
    )

    module = None
    if profile_runner is not None:
        from importlib import import_module

        module = import_module(
            "vllm.v1.worker.gpu_model_runner"
            if profile_runner == "v1"
            else "vllm.v1.worker.gpu.model_runner"
        )
    log: list[tuple[Any, ...]] = []
    calls: list[tuple[Any, str, int, int]] = []
    streams: list[Any] = []

    class Stream:
        def __init__(self, name):
            self.name = name

        def wait_stream(self, stream):
            log.append((self.name, "after", stream.name))

        def wait_event(self, event):
            log.append((self.name, "join", event.stream.name))

        def synchronize(self):
            log.append((self.name, "synchronize"))

    class Event:
        def record(self, stream):
            self.stream = stream
            log.append((stream.name, "event"))

        def synchronize(self):
            pass

    main = Stream("main")
    active = [main]

    def make_stream(**kwargs):
        stream = Stream(f"side-{len(streams)}")
        streams.append(stream)
        return stream

    @contextmanager
    def use_stream(stream):
        previous = active[0]
        active[0] = stream
        try:
            yield
        finally:
            active[0] = previous

    monkeypatch.setattr(torch.cuda, "current_stream", lambda: active[0])
    monkeypatch.setattr(torch.cuda, "Stream", make_stream)
    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "stream", use_stream)
    default, side_group = object(), object()
    monkeypatch.setattr(b12x_mla_sparse, "get_dcp_group", lambda: default)
    group_lookups = []

    def get_side_group():
        group_lookups.append(True)
        return side_group

    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_dcp_ckv_prefetch_group", get_side_group
    )

    def exchange(group, source, output):
        assert source.dtype == output.dtype == torch.uint8
        assert source.is_contiguous() and output.is_contiguous()
        assert output.numel() == source.numel() * 2
        calls.append((group, active[0].name, source.numel(), output.data_ptr()))
        output.view(-1).copy_(source.view(-1).repeat(2))

    monkeypatch.setattr(b12x_mla_sparse, "_dcp_all_gather_current_stream", exchange)
    plan = CKVPrefetchPlan.create(
        requested_depth=depth,
        budget_bytes=0,
        dcp_world_size=2,
        local_capacity=8,
        record_bytes=8,
        num_ubatches=2,
        num_lanes=2,
    )
    pool = CKVWorkspacePool(plan, torch.device("cpu"))
    reservation = b12x_mla_sparse._CKVReservation(
        CKVPrefetchRegistry(pool),
        torch.empty((2, 2, 3 if depth else 0, 8), dtype=torch.uint8),
        torch.empty((2, 2, 6 if depth else 0, 8), dtype=torch.uint8),
    )
    impl = object.__new__(B12xMLASparseImpl)
    impl._ckv_reservation = reservation
    impl._kernel_page_size = 4
    impl._cache_record_bytes = 8
    if profile_runner is None:
        impl.prepare_profile_collectives()
    else:
        assert module is not None
        runner = object.__new__(module.GPUModelRunner)
        runner.model_config = SimpleNamespace(architecture="GlmMoeDsaForCausalLM")
        runner.compilation_config = SimpleNamespace(
            static_forward_context={"attention": SimpleNamespace(impl=impl)}
        )
        runner.vllm_config = object()
        runner.max_num_tokens = 8
        runner.dcp_world_size = runner.dcp_size = 2
        runner.cp_interleave = 1
        profile_events = []

        def init_cache():
            assert not reservation.collectives_warmed
            profile_events.append("cache-bound")

        def gather(cache, metadata, local, output, **kwargs):
            output.zero_()
            return output

        impl._gather_full_ckv = gather

        def dummy_run(*args, **kwargs):
            # Exercise the serving guard and real registry/event lifecycle.
            cache = torch.zeros((2, 4, 8), dtype=torch.uint8)
            _, state, layer_index = impl._consume_ckv(
                cache,
                SimpleNamespace(),
                SimpleNamespace(layer_name="model.layers.0.self_attn"),
                cache,
            )
            completion = Event()
            completion.record(main)
            state.finish_consumer(layer_index, completion)
            profile_events.append("attention-consumed")
            return torch.empty(0), torch.empty(0)

        def cleanup():
            reservation.registry.clear()
            profile_events.append("cache-released")

        runner._dummy_run = dummy_run
        if profile_runner == "v1":
            runner._init_minimal_kv_cache_for_profiling = init_cache
            runner._cleanup_profiling_kv_cache = cleanup
            runner._sync_device = lambda: None
            monkeypatch.setattr(
                module, "set_current_vllm_config", lambda _: nullcontext()
            )
        else:
            monkeypatch.setattr(
                module, "_init_minimal_kv_cache_for_profiling", lambda _: init_cache()
            )
            monkeypatch.setattr(
                module, "_teardown_profiling_state", lambda _: cleanup()
            )
            monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)
        runner.profile_glm_dcp_attention()
        assert profile_events == ["cache-bound", "attention-consumed", "cache-released"]

    expected_history = {16, 32, 64}
    default_history = [c for c in calls if c[0] is default and c[2] in expected_history]
    assert len(default_history) == 4 * plan.ring_slots * 3
    assert {c[1] for c in default_history} == {"main"}
    assert len({c[3] for c in default_history}) == 4 * plan.ring_slots
    side_calls = [c for c in calls if c[0] is side_group]
    assert len(side_calls) == (4 * plan.ring_slots * 3 if depth else 0)
    assert {c[2] for c in side_calls} == (expected_history if depth else set())
    assert len([c for c in calls if c[2] == 24]) == (4 if depth else 0)
    assert len(streams) == (4 if depth else 0)
    assert group_lookups == ([True] if depth else [])
    assert ("main", "synchronize") in log
    assert reservation.collectives_warmed
    assert reservation.registry.states == {}
    if depth:
        assert (
            sum(event[1] == "join" and event[2].startswith("side-") for event in log)
            == 4
        )
        assert set(reservation.gather_streams) == {(0, 0), (0, 1), (1, 0), (1, 1)}
    before = len(calls), len(streams)
    reservation.registry.clear()
    impl._warmup_ckv_collectives()
    assert (len(calls), len(streams)) == before


def test_full_ckv_requires_collective_warmup_before_serving():
    impl = object.__new__(B12xMLASparseImpl)
    impl._ckv_reservation = SimpleNamespace(collectives_warmed=False)
    with pytest.raises(RuntimeError, match="warmup must finish before KV admission"):
        impl._consume_ckv(torch.empty(0), SimpleNamespace(), SimpleNamespace(), None)


@pytest.mark.parametrize("pynccl_disabled", [False, True])
def test_full_ckv_allgather_uses_effective_communicator(monkeypatch, pynccl_disabled):
    calls = []
    process_group = object()

    def pynccl(output, source):
        calls.append("pynccl")
        output.copy_(source.repeat(2))

    def torch_dist(output, source, *, group, async_op):
        assert group is process_group and not async_op
        calls.append("torch_dist")
        output.copy_(source.repeat(2))

    group = SimpleNamespace(
        world_size=2,
        device_group=process_group,
        device_communicator=SimpleNamespace(
            pynccl_comm=SimpleNamespace(disabled=pynccl_disabled, all_gather=pynccl)
        ),
    )
    monkeypatch.setattr(b12x_mla_sparse.dist, "all_gather_into_tensor", torch_dist)
    source = torch.arange(8, dtype=torch.uint8)
    output = torch.empty(16, dtype=torch.uint8)
    b12x_mla_sparse._dcp_all_gather_current_stream(group, source, output)
    assert calls == (["torch_dist"] if pynccl_disabled else ["pynccl"])
    assert torch.equal(output, source.repeat(2))


@pytest.mark.parametrize("depth", [0, 1])
@pytest.mark.parametrize("has_history", [False, True])
def test_full_ckv_prefetch_backend_inserts_changed_chunks_and_reuses_cache_identity(
    monkeypatch: pytest.MonkeyPatch, depth: int, has_history: bool
) -> None:
    from vllm.v1.attention.backends.mla.ckv_prefetch import (
        CKVPrefetchPlan,
        CKVPrefetchRegistry,
        CKVWorkspacePool,
    )

    log: list[tuple[Any, ...]] = []

    class Event:
        def record(self, stream):
            log.append(("record", stream.name))

        def synchronize(self):
            log.append(("synchronize",))

    class Stream:
        def __init__(self, name):
            self.name = name

        def wait_event(self, event):
            log.append(("wait", self.name))

        def wait_stream(self, stream):
            log.append(("producer", self.name, stream.name))

    main, side = Stream("main"), Stream("side")
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: main)
    monkeypatch.setattr(torch.cuda, "Stream", lambda **_: side)
    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "stream", lambda _: nullcontext())
    group = object()
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_dcp_ckv_prefetch_group", lambda: group
    )
    plan = CKVPrefetchPlan.create(
        requested_depth=depth,
        budget_bytes=0,
        dcp_world_size=2,
        local_capacity=4,
        record_bytes=656,
    )
    pool = CKVWorkspacePool(plan, torch.device("cpu"))
    reservation = b12x_mla_sparse._CKVReservation(
        CKVPrefetchRegistry(pool),
        torch.empty((1, 1, 4, 656), dtype=torch.uint8),
        torch.empty((1, 1, 8, 656), dtype=torch.uint8),
    )
    reservation.collectives_warmed = True
    if depth:
        reservation.gather_streams[(0, 0)] = side
    impl = object.__new__(B12xMLASparseImpl)
    impl._ckv_reservation = reservation
    impl._kernel_page_size = 2
    impl._cache_record_bytes = 656
    gathers, appends = [], []

    def gather(cache, metadata, local, output, *, group=None, history_only=False):
        assert history_only is (group is not None)
        gathers.append(group)
        output.fill_(int(cache.view(torch.uint8).flatten()[0]))
        return output.view(-1, 2, 656)

    def append(output, metadata, original_cache, lane):
        appends.append(original_cache)
        output.copy_(original_cache.repeat(2, 1, 1))

    impl._gather_full_ckv = gather
    impl._append_ckv_current_chunk = append
    caches = [torch.full((2, 2, 656), i, dtype=torch.uint8) for i in range(3)]
    metadata = SimpleNamespace(ckv_has_history=has_history)
    selections = []

    @contextmanager
    def scope(operation, **fields):
        selections.append(fields)
        yield

    impl._prefill_trace = SimpleNamespace(
        ownership_active=True,
        scope=scope,
        record=lambda event, stream: event.record(stream),
    )
    forced_stale = int(depth > 0 and not has_history)
    for execution in range(2):
        for layer_idx, cache in enumerate(caches):
            value = execution * 10 + layer_idx
            cache.fill_(value)
            layer = SimpleNamespace(layer_name=f"model.layers.{layer_idx}.self_attn")
            output, state, index = impl._consume_ckv(
                cache.view(torch.float8_e4m3fn), metadata, layer, cache
            )
            assert output.flatten()[0] == value
            assert state.layer_caches[layer_idx] is cache
            state.finish_consumer(index, Event())
            if forced_stale and execution == 1 and layer_idx == 0:
                impl._queue_ckv_gather(
                    state, 1, caches[1], metadata, side, main, asynchronous=True
                )
                assert 1 in state.pending
            elif not has_history:
                assert not state.pending
    assert all(fields["cache_identity_known"] == 1 for fields in selections)
    assert all(fields["has_history"] == int(has_history) for fields in selections)
    assert len(appends) == (2 if depth and has_history else 0)
    if depth and has_history:
        assert appends[0] is caches[1] and appends[1] is caches[2]
    assert sum(item is group for item in gathers) == (
        2 if depth and has_history else forced_stale
    )
    assert sum(item is None for item in gathers) == (4 if depth and has_history else 6)
    if forced_stale:
        assert ("wait", "main") in log
    assert ("producer", "side", "main") in log if depth else True
    reservation.registry.clear()


@pytest.mark.parametrize("capacity,depth", [(8, 1), (256, 0)])
def test_full_ckv_reservation_is_shared_and_charged_for_every_lane(
    monkeypatch, capacity, depth
) -> None:
    from vllm.v1.worker.workspace import WorkspaceManager

    manager = WorkspaceManager(torch.device("cpu"), num_ubatches=2, num_lanes=2)
    monkeypatch.setattr(b12x_mla_sparse, "current_workspace_manager", lambda: manager)
    monkeypatch.setenv("VLLM_B12X_MLA_CKV_PREFETCH_DEPTH", "1")
    monkeypatch.setenv("VLLM_B12X_MLA_CKV_PREFETCH_WORKSPACE_MIB", "1")
    communicator_calls = []
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.ensure_dcp_ckv_prefetch_group",
        lambda: communicator_calls.append(True),
        raising=False,
    )
    implementations = []
    for _ in range(2):
        impl = object.__new__(B12xMLASparseImpl)
        impl._is_glm_dsa = True
        impl.dcp_world_size = 4
        impl._ckv_local_capacity = capacity
        impl._ckv_current_capacity = 8
        impl._cache_record_bytes = 656
        impl._max_tokens = 32
        impl._decode_plan = SimpleNamespace(
            caps=SimpleNamespace(device=torch.device("cpu"))
        )
        impl._reserve_ckv_prefetch()
        implementations.append(impl)
    first, second = implementations
    assert first._ckv_reservation is second._ckv_reservation
    reservation = first._ckv_reservation
    assert reservation.registry.pool.plan.effective_depth == depth
    assert (
        reservation.registry.pool.storage.numel()
        == 4 * (1 + 4 * (depth + 1)) * capacity * 656
    )
    assert reservation.current_local.shape == (2, 2, 8 if depth else 0, 656)
    assert reservation.current_gathered.shape == (2, 2, 32 if depth else 0, 656)
    assert communicator_calls == ([True] if depth else [])
    first.reset_kv_cache_binding_state()


def test_full_ckv_reservation_gives_the_single_layer_drafter_lane_no_lookahead(
    monkeypatch,
) -> None:
    from vllm.v1.worker.workspace import WorkspaceManager

    manager = WorkspaceManager(torch.device("cpu"), num_ubatches=1, num_lanes=2)
    monkeypatch.setattr(b12x_mla_sparse, "current_workspace_manager", lambda: manager)
    monkeypatch.setenv("VLLM_B12X_MLA_CKV_PREFETCH_DEPTH", "1")
    monkeypatch.setenv("VLLM_B12X_MLA_CKV_PREFETCH_WORKSPACE_MIB", "0")
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.ensure_dcp_ckv_prefetch_group",
        lambda: None,
        raising=False,
    )
    impl = object.__new__(B12xMLASparseImpl)
    impl._is_glm_dsa = True
    impl.dcp_world_size = 4
    impl._ckv_local_capacity = 8
    impl._ckv_current_capacity = 8
    impl._cache_record_bytes = 656
    impl._max_tokens = 32
    impl._ckv_lane_layer_counts = (78, 1)
    impl._decode_plan = SimpleNamespace(
        caps=SimpleNamespace(device=torch.device("cpu"))
    )
    impl._reserve_ckv_prefetch()
    plan = impl._ckv_reservation.registry.pool.plan
    assert plan.lane_depths == (1, 0) and plan.effective_depth == 1
    # Target lane: staging plus two ring slots; drafter lane: staging plus one.
    assert (
        impl._ckv_reservation.registry.pool.storage.numel()
        == ((1 + 4 * 2) + (1 + 4 * 1)) * 8 * 656
    )
    impl.reset_kv_cache_binding_state()


def test_ckv_lane_layer_counts_read_target_and_drafter_layers() -> None:
    target = SimpleNamespace(num_hidden_layers=78)
    draft = SimpleNamespace(num_hidden_layers=78, num_nextn_predict_layers=1)
    config = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=target),
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(hf_text_config=draft)
        ),
    )
    assert b12x_mla_sparse._ckv_lane_layer_counts(config) == (78, 1)
    config.speculative_config = None
    assert b12x_mla_sparse._ckv_lane_layer_counts(config) == (78, 78)
    config.speculative_config = SimpleNamespace(
        draft_model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(num_hidden_layers=2)
        )
    )
    assert b12x_mla_sparse._ckv_lane_layer_counts(config) == (78, 2)


@pytest.mark.parametrize("record_bytes", [656, 368])
def test_full_ckv_current_chunk_copies_producer_bytes_without_requantization(
    monkeypatch, record_bytes
) -> None:
    from b12x.attention._shared.mla import kv_cache as native

    impl = object.__new__(B12xMLASparseImpl)
    impl.dcp_world_size = 4
    impl.dcp_rank = 0
    impl._ckv_current_capacity = 3
    local = torch.empty((1, 1, 3, record_bytes), dtype=torch.uint8)
    current = torch.empty((1, 1, 12, record_bytes), dtype=torch.uint8)
    impl._ckv_reservation = b12x_mla_sparse._CKVReservation(None, local, current)
    metadata = SimpleNamespace(
        query_start_loc=torch.tensor([0, 1, 3], dtype=torch.int32),
        global_cache_seq_lens_per_req=torch.tensor([17, 9], dtype=torch.int32),
        dcp_rank_req_starts=torch.zeros((4, 2), dtype=torch.int32),
        block_table=torch.tensor([[0], [1]], dtype=torch.int32),
        num_reqs=2,
        dcp_padded_total_tokens=16,
        cp_kv_cache_interleave_size=4,
    )
    gathered = torch.empty((4, 16, record_bytes), dtype=torch.uint8)
    calls = []

    def pack(source, output, blocks, lengths, starts, **kwargs):
        assert source.dtype == torch.uint8
        assert blocks is metadata.block_table
        assert lengths is metadata.global_cache_seq_lens_per_req
        assert starts is metadata.query_start_loc
        assert kwargs == dict(
            dcp_rank=0, dcp_world_size=4, interleave=4, num_reqs=2, current_capacity=3
        )
        output.copy_(source.view(3, record_bytes))
        calls.append("pack")

    def exchange(group, source, output):
        output.copy_(source.repeat(4, 1))
        calls.append("exchange")

    def insert(source, output, rank_starts, lengths, starts, **kwargs):
        assert output is gathered
        assert rank_starts is metadata.dcp_rank_req_starts
        assert kwargs["current_capacity"] == 3
        assert kwargs["padded_tokens"] == 16
        output.view(-1, record_bytes)[:12].copy_(source)
        calls.append("insert")

    monkeypatch.setattr(native, "gather_ckv_current_chunk", pack)
    monkeypatch.setattr(native, "insert_ckv_current_chunk", insert)
    monkeypatch.setattr(b12x_mla_sparse, "_dcp_all_gather_current_stream", exchange)
    monkeypatch.setattr(b12x_mla_sparse, "get_dcp_group", lambda: object())
    for value in (1, 199):
        producer_bytes = torch.full((1, 3, record_bytes), value, dtype=torch.uint8)
        impl._append_ckv_current_chunk(gathered, metadata, producer_bytes, (0, 0))
        assert torch.all(gathered.view(-1, record_bytes)[:12] == value)
    assert calls == ["pack", "exchange", "insert"] * 2


@pytest.mark.parametrize("override", [{"enabled": False}, {"dcp_world_size": 1}])
def test_b12x_full_ckv_gather_requires_enabled_dcp(override: dict[str, Any]) -> None:
    args = dict(
        enabled=True,
        is_glm_next=True,
        dcp_world_size=4,
        max_query_len=128,
        num_tokens=128,
        num_decode_tokens=0,
        min_tokens=16,
        max_tokens=524288,
    )
    args.update(override)

    assert not _use_b12x_full_ckv_gather(**args)


@pytest.mark.parametrize("dcp_size,enabled", [(1, True), (4, False), (4, True)])
@pytest.mark.parametrize("local_heads,enable_dbo", [(8, False), (4, False), (8, True)])
def test_b12x_glm_dsa_full_ckv_builder_allocates_exact_selector_width(
    monkeypatch: pytest.MonkeyPatch,
    dcp_size: int,
    enabled: bool,
    local_heads: int,
    enable_dbo: bool,
) -> None:
    config = _glm5_next_config(dcp_size=dcp_size)
    config.model_config.hf_text_config.model_type = "glm_moe_dsa"
    del config.model_config.hf_text_config.index_kpool
    config.model_config.get_num_attention_heads = lambda _: local_heads
    config.parallel_config.enable_dbo = enable_dbo
    config.scheduler_config = SimpleNamespace(max_num_batched_tokens=32, max_num_seqs=2)
    monkeypatch.setenv("VLLM_B12X_MLA_CKV_GATHER", str(int(enabled)))

    def initialize_common(builder, spec, layer_names, config, device):
        builder.dcp_world_size = dcp_size
        builder.kv_cache_spec = spec
        builder.cp_kv_cache_interleave_size = 4

    monkeypatch.setattr(SparseMLACommonMetadataBuilder, "__init__", initialize_common)
    monkeypatch.setattr(
        B12xMLASparseMetadataBuilder,
        "_init_reorder_batch_threshold",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        b12x_mla_sparse, "get_dcp_group", lambda: SimpleNamespace(rank_in_group=0)
    )
    builder = B12xMLASparseMetadataBuilder(
        SimpleNamespace(block_size=64), [], config, torch.device("cpu")
    )

    assert not builder.requires_glm_next_selector_metadata
    assert not builder.supports_draft_decode_metadata_update
    assert builder._ckv_gather_requested is (
        enabled and dcp_size > 1 and local_heads == 8 and not enable_dbo
    )
    if builder._ckv_gather_requested:
        assert builder.ckv_selected_indices_buffer.shape == (32, 2048)
        assert builder.dcp_rank_req_lens_buffer.shape == (4, 2)
    else:
        assert builder.ckv_selected_indices_buffer is None
        assert builder.dcp_rank_req_lens_buffer is None

    query_starts = torch.tensor([0, 16, 32], dtype=torch.int32)
    seq_lens = torch.tensor([19, 33], dtype=torch.int32)
    common = SimpleNamespace(
        num_reqs=2,
        num_actual_tokens=32,
        max_query_len=16,
        seq_lens=seq_lens,
        seq_lens_cpu=seq_lens,
        seq_lens_cpu_upper_bound=None,
        query_start_loc=query_starts,
        query_start_loc_cpu=query_starts,
        dcp_local_seq_lens=None,
        positions=torch.cat((torch.arange(3, 19), torch.arange(17, 33))),
        is_prefilling=torch.tensor([True, True]),
    )
    monkeypatch.setattr(
        SparseMLACommonMetadataBuilder,
        "build",
        lambda *a, **k: SimpleNamespace(
            num_prefills=2,
            num_decodes=0,
            num_decode_tokens=0,
            num_actual_tokens=32,
            dcp_ckv_gather_eligible=False,
            ckv_has_history=True,
        ),
    )
    if not builder._ckv_gather_requested:

        def unexpected_equal(*args, **kwargs):
            pytest.fail("Disabled CKV must not compare CPU history lengths")

        with monkeypatch.context() as patch:
            patch.setattr(torch, "equal", unexpected_equal)
            metadata = builder.build(0, common)
        assert metadata.ckv_has_history
        assert not metadata.dcp_ckv_gather_eligible
        return
    metadata = builder.build(0, common)
    assert metadata.dcp_ckv_gather_eligible
    assert metadata.ckv_selected_indices.shape == (32, 2048)
    assert metadata.dcp_rank_req_lens.tolist() == [[7, 9], [4, 8], [4, 8], [4, 8]]
    assert metadata.dcp_rank_req_starts.tolist() == [[0, 7], [0, 4], [0, 4], [0, 4]]
    assert metadata.dcp_local_cu_seq_lens.tolist() == [0, 7, 16]
    assert metadata.dcp_padded_total_tokens == 16
    assert metadata.global_cache_seq_lens_per_req.tolist() == [19, 33]
    assert metadata.ckv_has_history
    for lengths, upper_bound, expected in (
        ([16, 16], None, False),  # Entirely initial chunks.
        ([32, 32], None, True),  # Later chunks.
        ([80, 16], None, True),  # Reused prefix in one request.
        ([16, 17], None, True),  # A single history token is sufficient.
        ([16, 16], [17, 16], True),  # Uncertain bound must not skip history.
        ([16, 16], [16, 16], False),
    ):
        common.seq_lens = torch.tensor(lengths, dtype=torch.int32)
        common.seq_lens_cpu = common.seq_lens
        common.seq_lens_cpu_upper_bound = (
            None
            if upper_bound is None
            else torch.tensor(upper_bound, dtype=torch.int32)
        )
        for rank in range(dcp_size):
            builder.dcp_rank = rank
            assert builder.build(0, common).ckv_has_history is expected
    captured = builder._build(0, common, for_cudagraph_capture=True)
    assert not captured.dcp_ckv_gather_eligible


@pytest.mark.parametrize(
    "is_spec_decode,num_decode_tokens", [(False, 0), (False, 1), (True, 0)]
)
def test_b12x_glm_dsa_full_ckv_excludes_mixed_and_verification(
    is_spec_decode: bool, num_decode_tokens: int
) -> None:
    assert _use_b12x_full_ckv_gather(
        enabled=True,
        is_glm_next=False,
        is_glm_dsa=True,
        is_spec_decode=is_spec_decode,
        dcp_world_size=4,
        max_query_len=4,
        num_tokens=32,
        num_decode_tokens=num_decode_tokens,
        min_tokens=16,
        max_tokens=524288,
    ) is (not is_spec_decode and num_decode_tokens == 0)


def test_b12x_glm5_next_accepts_dcp_with_prefix_caching(monkeypatch) -> None:
    monkeypatch.setattr(b12x_mla_sparse, "get_b12x_sparse_mla", lambda: object())
    with set_current_vllm_config(
        _glm5_next_config(
            dcp_size=4,
            cp_interleave=4,
            prefix_caching=True,
        )
    ):
        invalid_reasons = B12xMLASparseBackend.validate_configuration(
            head_size=512,
            dtype=torch.bfloat16,
            kv_cache_dtype="fp8",
            block_size=64,
            use_mla=True,
            has_sink=False,
            use_sparse=True,
            use_mm_prefix=False,
            use_per_head_quant_scales=False,
            device_capability=DeviceCapability(12, 0),
            attn_type="decoder",
        )

    assert invalid_reasons == []


def test_b12x_glm5_next_rejects_dsv4_head_size(monkeypatch) -> None:
    monkeypatch.setattr(b12x_mla_sparse, "get_b12x_sparse_mla", lambda: object())
    with set_current_vllm_config(_glm5_next_config()):
        invalid_reasons = B12xMLASparseBackend.validate_configuration(
            head_size=576,
            dtype=torch.bfloat16,
            kv_cache_dtype="fp8",
            block_size=64,
            use_mla=True,
            has_sink=False,
            use_sparse=True,
            use_mm_prefix=False,
            use_per_head_quant_scales=False,
            device_capability=DeviceCapability(12, 0),
            attn_type="decoder",
        )

    assert invalid_reasons == ["B12X GLM5Next sparse MLA requires head_size=512"]


def test_b12x_glm5_next_rejects_recipe_drift(monkeypatch) -> None:
    monkeypatch.setattr(b12x_mla_sparse, "get_b12x_sparse_mla", lambda: object())
    with set_current_vllm_config(_glm5_next_config(index_kpool=8)):
        invalid_reasons = B12xMLASparseBackend.validate_configuration(
            head_size=512,
            dtype=torch.bfloat16,
            kv_cache_dtype="fp8",
            block_size=64,
            use_mla=True,
            has_sink=False,
            use_sparse=True,
            use_mm_prefix=False,
            use_per_head_quant_scales=False,
            device_capability=DeviceCapability(12, 0),
            attn_type="decoder",
        )

    assert invalid_reasons == [
        "B12X GLM5Next sparse MLA requires index_kpool=8 (expected 4)"
    ]


def test_b12x_glm5_next_ckv_source_layout() -> None:
    storage = torch.empty((2 * 37888,), dtype=torch.uint8)
    cache = torch.as_strided(
        storage,
        size=(2, 64, 528),
        stride=(37888, 528, 1),
    )
    assert _is_native_ckv_source_layout(cache, page_size=64, record_bytes=528)
    assert not _is_native_ckv_source_layout(
        cache[:, :, ::2], page_size=64, record_bytes=528
    )


@pytest.mark.parametrize("record_bytes", [528, 304])
def test_b12x_glm5_next_full_ckv_workspaces_follow_cache_format(
    record_bytes: int,
) -> None:
    impl = object.__new__(B12xMLASparseImpl)
    impl._max_tokens = 32
    impl._q_head_dim = 512
    impl._scratch_nbytes = 16
    impl._ckv_local_capacity = 128
    impl.dcp_world_size = 4
    impl._cache_record_bytes = record_bytes
    plan = object()

    specs = impl._workspace_specs(plan, input_num_heads=8, include_ckv=True)

    assert specs[-2:] == (
        ((128, record_bytes), torch.uint8),
        ((512, record_bytes), torch.uint8),
    )


@pytest.mark.parametrize(
    "record_bytes,input_dtype",
    [
        (528, torch.uint8),
        (304, torch.uint8),
        (656, torch.uint8),
        (368, torch.uint8),
        (656, torch.float8_e4m3fn),
    ],
)
def test_b12x_full_ckv_gather_preserves_native_records(
    monkeypatch: pytest.MonkeyPatch,
    record_bytes: int,
    input_dtype: torch.dtype,
) -> None:
    impl = object.__new__(B12xMLASparseImpl)
    impl._kernel_page_size = 2
    impl._ckv_local_capacity = 4
    impl._cache_record_bytes = record_bytes
    impl.dcp_world_size = 2
    impl.uses_full_ckv_dcp = lambda *_: True

    kv_cache = (
        torch.arange(4 * record_bytes, dtype=torch.int64)
        .to(torch.uint8)
        .view(2, 2, record_bytes)
    )
    local_buffer = torch.full((4, record_bytes), 255, dtype=torch.uint8)
    gathered_buffer = torch.empty((8, record_bytes), dtype=torch.uint8)
    metadata = SimpleNamespace(
        num_actual_tokens=3,
        dcp_local_total_tokens=3,
        dcp_padded_total_tokens=4,
        dcp_local_cu_seq_lens=torch.tensor([0, 3], dtype=torch.int32),
        block_table=torch.tensor([[0, 1]], dtype=torch.int32),
        num_reqs=1,
    )

    def fake_cp_gather_cache(**kwargs: Any) -> None:
        assert kwargs["src_cache"].dtype == torch.uint8
        kwargs["dst"].copy_(kwargs["src_cache"].view(-1, record_bytes)[:3])

    def fake_all_gather(_group: Any, src: torch.Tensor, dst: torch.Tensor) -> None:
        dst.copy_(src.repeat(2))

    monkeypatch.setattr(b12x_mla_sparse.ops, "cp_gather_cache", fake_cp_gather_cache)
    monkeypatch.setattr(
        b12x_mla_sparse, "_dcp_all_gather_current_stream", fake_all_gather
    )
    monkeypatch.setattr(b12x_mla_sparse, "get_dcp_group", lambda: object())

    gathered = impl._gather_full_ckv(
        kv_cache.view(input_dtype), metadata, local_buffer, gathered_buffer
    )

    expected_rank = torch.cat(
        (kv_cache.view(-1, record_bytes)[:3], torch.zeros((1, record_bytes))),
        dim=0,
    )
    assert gathered.shape == (4, 2, record_bytes)
    assert torch.equal(gathered.view(-1, record_bytes), expected_rank.repeat(2, 1))


@pytest.mark.parametrize("record_bytes", [656, 368])
def test_b12x_full_ckv_gather_reuses_capacity_with_changed_request_records(
    monkeypatch: pytest.MonkeyPatch,
    record_bytes: int,
) -> None:
    """Backend borrows stable storage while copies observe each batch's records."""
    impl = object.__new__(B12xMLASparseImpl)
    impl._kernel_page_size = 2
    impl._ckv_local_capacity = 4
    impl._cache_record_bytes = record_bytes
    impl.dcp_world_size = 2
    impl.uses_full_ckv_dcp = lambda *_: True
    cache = torch.empty((4, 2, record_bytes), dtype=torch.uint8)
    local = torch.full((4, record_bytes), 255, dtype=torch.uint8)
    gathered = torch.full((8, record_bytes), 255, dtype=torch.uint8)
    storage_ptr = gathered.data_ptr()
    block_table = torch.tensor([[2, 0], [3, 1]], dtype=torch.int32)
    copies = []

    def copy_requests(**kwargs: Any) -> None:
        assert kwargs["block_table"] is block_table
        assert kwargs["batch_size"] == 2
        starts = kwargs["cu_seq_lens"].tolist()
        copies.append(starts)
        for request in range(2):
            for offset in range(starts[request + 1] - starts[request]):
                page = int(block_table[request, offset // 2])
                kwargs["dst"][starts[request] + offset].copy_(
                    kwargs["src_cache"][page, offset % 2]
                )

    def gather_ranks(_group: Any, src: torch.Tensor, dst: torch.Tensor) -> None:
        dst.copy_(src.repeat(2))

    monkeypatch.setattr(b12x_mla_sparse.ops, "cp_gather_cache", copy_requests)
    monkeypatch.setattr(b12x_mla_sparse, "_dcp_all_gather_current_stream", gather_ranks)
    monkeypatch.setattr(b12x_mla_sparse, "get_dcp_group", lambda: object())

    for generation, (lengths, physical_slots) in enumerate(
        [([0, 1], [6]), ([2, 1], [4, 5, 6]), ([1, 3], [4, 6, 7, 2]), ([0, 0], [])]
    ):
        for slot in range(8):
            cache.view(8, record_bytes)[slot].fill_(generation * 16 + slot)
        token_count = sum(lengths)
        padded = max(1, token_count)
        metadata = SimpleNamespace(
            num_actual_tokens=2,
            dcp_local_total_tokens=token_count,
            dcp_padded_total_tokens=padded,
            dcp_local_cu_seq_lens=torch.tensor(
                [0, lengths[0], token_count], dtype=torch.int32
            ),
            block_table=block_table,
            num_reqs=2,
        )

        result = impl._gather_full_ckv(cache, metadata, local, gathered)

        expected_rank = torch.zeros((padded, record_bytes), dtype=torch.uint8)
        for row, slot in enumerate(physical_slots):
            expected_rank[row].fill_(generation * 16 + slot)
        assert result.shape == (4, 2, record_bytes)
        assert result.data_ptr() == storage_ptr
        assert torch.equal(gathered[: 2 * padded], expected_rank.repeat(2, 1))

    assert copies == [[0, 0, 1], [0, 2, 3], [0, 1, 4]]


def test_b12x_glm5_next_full_ckv_gather_rejects_wrong_record_width() -> None:
    impl = object.__new__(B12xMLASparseImpl)
    impl._kernel_page_size = 2
    impl._ckv_local_capacity = 4
    impl._cache_record_bytes = 304
    impl.dcp_world_size = 2
    impl.uses_full_ckv_dcp = lambda *_: True
    metadata = SimpleNamespace(num_actual_tokens=1)

    with pytest.raises(ValueError, match="requires native 304-byte records"):
        impl._gather_full_ckv(
            torch.empty((2, 2, 528), dtype=torch.uint8),
            metadata,
            torch.empty((4, 304), dtype=torch.uint8),
            torch.empty((8, 304), dtype=torch.uint8),
        )


@pytest.mark.parametrize(
    ("page_size", "dcp_world_size", "alignment"),
    [(2048, 4, 512), (2048, 2, 1024), (512, 4, 128), (512, 3, 512)],
)
def test_full_ckv_rank_alignment_only_pads_the_concatenated_cache_to_pages(
    page_size: int,
    dcp_world_size: int,
    alignment: int,
) -> None:
    assert _ckv_rank_token_alignment(page_size, dcp_world_size) == alignment
    padded = _round_up_ckv_rank_tokens(
        1025,
        page_size=page_size,
        dcp_world_size=dcp_world_size,
    )
    assert padded >= 1025
    assert padded % alignment == 0
    assert padded * dcp_world_size % page_size == 0


@pytest.mark.parametrize("record_bytes", [528, 656])
def test_b12x_selected_indices_use_physical_slots(record_bytes: int) -> None:
    storage = torch.empty((2, 2, 64, record_bytes), dtype=torch.uint8)
    cache = storage[:, 0]

    assert cache.stride(0) // record_bytes == 128
    assert _selected_index_block_stride_rows(cache, block_size=64) == 64


def test_sparse_index_remap_tiling_covers_glm5_next_width() -> None:
    assert _remap_tiling(2048, 128, True) == (True, 2048, 1, 8)
    assert _remap_tiling(2051, 128, True) == (False, 128, 17, 4)


def test_b12x_glm5_next_cache_writer_ignores_empty_rope() -> None:
    calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    impl = object.__new__(B12xMLASparseImpl)
    impl._is_glm_next = True
    impl._concat_and_cache_glm_next_mla = lambda *args: calls.append(args)
    kv_c = torch.empty((3, 512), dtype=torch.bfloat16)
    kv_cache = torch.empty((2, 64, 528), dtype=torch.uint8)
    slots = torch.tensor([0, 64, -1], dtype=torch.int64)

    impl.do_kv_cache_update(
        kv_c,
        torch.empty((3, 1, 0), dtype=torch.bfloat16),
        kv_cache,
        slots,
        "fp8_ds_mla",
        torch.ones((), dtype=torch.float32),
    )

    assert calls == [(kv_c, kv_cache, slots)]


def test_b12x_glm_dsa_nvfp4_cache_writer_keeps_rope() -> None:
    calls: list[tuple[torch.Tensor, ...]] = []
    impl = object.__new__(B12xMLASparseImpl)
    impl._is_glm_next = False
    impl._uses_glm_dsa_nvfp4_cache = True
    impl._concat_and_cache_nvfp4_mla_fp8_rope = lambda *args: calls.append(args)
    kv_c = torch.zeros((3, 512), dtype=torch.bfloat16)
    k_pe = torch.zeros((3, 1, 64), dtype=torch.bfloat16)
    kv_cache = torch.empty((2, 64, 368), dtype=torch.uint8)
    slots = torch.tensor([0, 64, -1], dtype=torch.int64)
    scale = torch.ones((), dtype=torch.float32)

    impl.do_kv_cache_update(
        kv_c,
        k_pe,
        kv_cache,
        slots,
        "nvfp4_ds_mla",
        scale,
    )

    assert len(calls) == 1
    actual_kv_c, actual_k_pe, actual_cache, actual_slots, actual_scale = calls[0]
    assert actual_kv_c is kv_c
    assert torch.equal(actual_k_pe, k_pe.squeeze(1))
    assert actual_cache is kv_cache
    assert torch.equal(actual_slots, slots)
    assert actual_scale is scale


def test_b12x_glm5_next_cache_geometry_is_finalized_before_bind(monkeypatch) -> None:
    planned: list[SimpleNamespace] = []
    reservations: list[tuple[tuple[tuple[int, ...], torch.dtype], ...]] = []
    persistent_geometry: list[tuple[int, int, int, bool]] = []
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)
    monkeypatch.setattr(
        b12x_mla_sparse,
        "is_workspace_manager_initialized",
        lambda: False,
    )
    monkeypatch.setattr(
        b12x_mla_sparse,
        "current_workspace_manager",
        lambda: SimpleNamespace(reserve_all=lambda *specs: reservations.append(specs)),
    )

    class FakeCaps(SimpleNamespace):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.cache_record_bytes = 528
            self.layout = SimpleNamespace(nbytes=256)

        def shapes_and_dtypes(self):
            return (((self.layout.nbytes,), torch.uint8),)

    class FakeModule:
        Caps = FakeCaps

        @staticmethod
        def plan(caps):
            planned.append(caps)
            return SimpleNamespace(
                caps=caps,
                layout=caps.layout,
                shapes_and_dtypes=caps.shapes_and_dtypes,
            )

    impl = object.__new__(B12xMLASparseImpl)
    impl._is_glm_next = True
    impl._uses_nvfp4_cache = False
    impl._cache_record_bytes = 528
    impl._module = FakeModule
    impl._kernel_page_size = 64
    impl._kernel_page_size_finalized = False
    impl._input_num_heads = 64
    impl.num_heads = 16
    impl.dcp_world_size = 4
    impl._max_tokens = 4096
    impl._max_seqs = 4
    impl._max_speculative_decode_query_len = 6
    impl._decode_max_rows = 24
    impl._topk_tokens = 2051
    impl._kv_dtype = torch.uint8
    impl._q_head_dim = 512
    impl.kv_lora_rank = 512
    impl.scale = 256**-0.5
    impl.need_to_return_lse_for_decode = True
    impl._model_type = 1
    impl._ckv_gather_enabled = True
    impl._ckv_capacity_tokens = 131200
    impl._ckv_local_capacity = 131200
    impl._ckv_interleave = 1
    impl._decode_plan = SimpleNamespace()
    impl._extend_plan = SimpleNamespace()
    impl._ckv_extend_plan = SimpleNamespace()
    impl._reserve_ckv_prefetch = lambda: persistent_geometry.append(
        (
            impl._kernel_page_size,
            impl._ckv_local_capacity,
            impl._cache_record_bytes,
            impl._kernel_page_size_finalized,
        )
    )
    owner = SimpleNamespace(impl=impl, indexer=None)
    cache = torch.empty((2, 1, 2304, 528), dtype=torch.uint8)

    MLAAttention.finalize_kv_cache_geometry(
        owner,
        SimpleNamespace(cache_config=SimpleNamespace(block_size=2304)),
    )
    MLAAttention.bind_kv_cache(owner, cache)

    assert owner.kv_cache.shape == (2, 2304, 528)
    assert impl._kernel_page_size == 2304
    assert impl._kernel_page_size_finalized
    assert [(caps.mode, caps.page_size) for caps in planned] == [
        ("decode", 2304),
        ("extend", 2304),
        ("extend", 2304),
    ]
    plan_geometry = [
        (caps.num_q_heads, caps.max_q_rows, caps.max_batch) for caps in planned
    ]
    assert plan_geometry == [
        (64, 24, 24),
        (64, 4096, 4096),
        (16, 4096, 4096),
    ]
    assert len(reservations) == 3
    assert reservations[0] == (
        ((4096, 64, 512), torch.bfloat16),
        ((256,), torch.uint8),
    )
    assert reservations[1] == (
        ((4096, 64, 512), torch.bfloat16),
        ((256,), torch.uint8),
    )
    assert reservations[2] == (
        ((4096, 16, 512), torch.bfloat16),
        ((256,), torch.uint8),
    )
    assert persistent_geometry == [(2304, 131328, 528, True)]

    with pytest.raises(RuntimeError, match="immutable after finalization"):
        impl.finalize_kv_cache_geometry(64)
    with pytest.raises(RuntimeError, match="does not match the finalized"):
        impl.bind_kv_cache(torch.empty((2, 64, 528), dtype=torch.uint8))


def test_b12x_glm5_next_full_ckv_bind_requires_geometry_finalization() -> None:
    impl = object.__new__(B12xMLASparseImpl)
    impl._is_glm_next = True
    impl._cache_record_bytes = 528
    impl._ckv_gather_enabled = True
    impl._kernel_page_size_finalized = False

    with pytest.raises(RuntimeError, match="before KV-cache memory profiling"):
        impl.bind_kv_cache(torch.empty((2, 2304, 528), dtype=torch.uint8))


@pytest.mark.parametrize(
    ("parallel_drafting", "expected_query_len"),
    [(False, 6), (True, 11)],
)
def test_b12x_sparse_mla_bounds_speculative_decode_query_len(
    parallel_drafting: bool,
    expected_query_len: int,
) -> None:
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            num_speculative_tokens=5,
            parallel_drafting=parallel_drafting,
        )
    )

    assert _max_speculative_decode_query_len(config) == expected_query_len


@pytest.mark.parametrize(
    ("max_query_len", "is_prefilling", "expected"),
    [
        (1, [False, False], False),
        (6, [False, False], True),
        (6, [False, True], False),
        (7, [False, False], False),
    ],
)
def test_b12x_sparse_mla_identifies_only_speculative_verifier_batches(
    max_query_len: int,
    is_prefilling: list[bool],
    expected: bool,
) -> None:
    common = SimpleNamespace(
        num_reqs=2,
        max_query_len=max_query_len,
        is_prefilling=torch.tensor(is_prefilling),
    )

    assert _is_speculative_decode_batch(common, 6) is expected


@pytest.mark.parametrize(
    ("max_query_len", "is_spec_decode", "num_tokens", "expected_decode"),
    [
        (1, False, 4, True),
        (6, True, 24, True),
        (6, False, 24, False),
        (6, True, 25, False),
        (7, True, 24, False),
    ],
)
def test_b12x_sparse_mla_routes_only_planned_decode_rows(
    max_query_len: int,
    is_spec_decode: bool,
    num_tokens: int,
    expected_decode: bool,
) -> None:
    impl = object.__new__(B12xMLASparseImpl)
    impl._max_speculative_decode_query_len = 6
    impl._decode_max_rows = 24
    metadata = SimpleNamespace(
        num_reqs=4,
        max_query_len=max_query_len,
        is_spec_decode=is_spec_decode,
    )

    assert impl._use_decode_plan(metadata, num_tokens) is expected_decode


def test_b12x_sparse_mla_reserves_largest_planned_workspace(monkeypatch) -> None:
    reservations: list[tuple[tuple[tuple[int, ...], torch.dtype], ...]] = []
    manager = SimpleNamespace(
        get_simultaneous=lambda *specs: reservations.append(specs)
    )
    monkeypatch.setattr(
        b12x_mla_sparse,
        "is_workspace_manager_initialized",
        lambda: True,
    )
    monkeypatch.setattr(
        b12x_mla_sparse,
        "current_workspace_manager",
        lambda: manager,
    )

    impl = object.__new__(B12xMLASparseImpl)
    impl._ckv_gather_enabled = False
    impl._max_tokens = 64
    impl._input_num_heads = 8
    impl._q_head_dim = 512
    impl._scratch_nbytes = 512
    impl._decode_plan = SimpleNamespace(
        shapes_and_dtypes=lambda: (((32,), torch.uint8),)
    )
    impl._extend_plan = SimpleNamespace(
        shapes_and_dtypes=lambda: (((512,), torch.uint8),)
    )

    impl._reserve_planned_workspaces()

    assert reservations == [
        (
            ((64, 8, 512), torch.bfloat16),
            ((512,), torch.uint8),
        )
    ]


def _bare_glm_selector_metadata_builder() -> B12xMLASparseMetadataBuilder:
    builder = B12xMLASparseMetadataBuilder.__new__(B12xMLASparseMetadataBuilder)
    builder.requires_glm_next_selector_metadata = True
    builder.supports_draft_decode_metadata_update = True
    builder._ckv_gather_requested = False
    builder.dcp_world_size = 1
    builder._max_speculative_decode_query_len = 6
    builder._capture_default_state_slot_ids = torch.arange(4, dtype=torch.int32)
    builder._capture_state_slot_ids = torch.empty(4, dtype=torch.int32)
    builder._capture_state_is_fresh = torch.ones(4, dtype=torch.bool)
    builder._capture_num_accepted_tokens = torch.ones(4, dtype=torch.int32)
    builder._capture_is_prefilling = torch.zeros(4, dtype=torch.bool)
    return builder


def _build_short_packed_metadata(
    builder_cls: type[B12xMLASparseMetadataBuilder],
    *,
    seq_lens: list[int],
    query_lens: list[int],
    is_prefilling: list[bool],
) -> B12xMLASparseMetadata:
    builder = object.__new__(builder_cls)
    builder.metadata_cls = B12xMLASparseMetadata
    builder.require_uniform_decodes = False
    builder.use_pcp = False
    builder.reorder_batch_threshold = 128
    builder._prefill_backend = None
    builder.topk_tokens = 2048
    builder.cp_kv_cache_interleave_size = 1
    builder.kv_cache_spec = SimpleNamespace(block_size=64)
    builder.model_config = SimpleNamespace(dtype=torch.bfloat16)
    rows = sum(query_lens)
    query_start_loc = torch.tensor(
        [0, *torch.tensor(query_lens).cumsum(0).tolist()],
        dtype=torch.int32,
    )
    request_ids = torch.repeat_interleave(
        torch.arange(len(query_lens), dtype=torch.int32),
        torch.tensor(query_lens),
    )
    builder._build_req_id_per_token = lambda common: request_ids
    positions = torch.cat(
        [torch.arange(length, dtype=torch.int64) for length in query_lens]
    )
    common = SimpleNamespace(
        num_reqs=len(seq_lens),
        num_actual_tokens=rows,
        max_query_len=max(query_lens),
        max_seq_len=max(seq_lens),
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc,
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
        block_table_tensor=torch.arange(len(seq_lens), dtype=torch.int32).view(-1, 1),
        slot_mapping=torch.arange(rows, dtype=torch.int64),
        positions=positions,
        is_prefilling=torch.tensor(is_prefilling),
    )
    return SparseMLACommonMetadataBuilder.build(builder, 0, common)


def test_glm_short_packed_prefills_do_not_use_selector_decode_transactions() -> None:
    fresh = _build_short_packed_metadata(
        B12xGLM5NextMLASparseMetadataBuilder,
        seq_lens=[2, 3],
        query_lens=[2, 3],
        is_prefilling=[True, True],
    )
    assert fresh.num_decodes == 0
    assert fresh.num_prefills == 2
    assert fresh.num_decode_tokens == 0
    assert fresh.req_id_per_token.tolist() == [0, 0, 1, 1, 1]
    assert fresh.query_start_loc.tolist() == [0, 2, 5]

    mixed = _build_short_packed_metadata(
        B12xGLM5NextMLASparseMetadataBuilder,
        seq_lens=[4, 2],
        query_lens=[1, 2],
        is_prefilling=[False, True],
    )
    assert mixed.num_decodes == 1
    assert mixed.num_prefills == 1
    assert mixed.num_decode_tokens == 1
    assert mixed.req_id_per_token.tolist() == [0, 1, 1]
    assert mixed.query_start_loc.tolist() == [0, 1, 3]

    dsv4 = _build_short_packed_metadata(
        B12xMLASparseMetadataBuilder,
        seq_lens=[2, 3],
        query_lens=[2, 3],
        is_prefilling=[True, True],
    )
    assert dsv4.num_decodes == 2
    assert dsv4.num_prefills == 0
    assert dsv4.num_decode_tokens == 5


def test_glm_selector_metadata_builder_stages_padded_rows_and_capture(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        SparseMLACommonMetadataBuilder,
        "build",
        lambda *args, **kwargs: SimpleNamespace(
            num_prefills=0,
            num_decode_tokens=0,
        ),
    )
    builder = _bare_glm_selector_metadata_builder()
    common = SimpleNamespace(
        num_reqs=4,
        num_actual_tokens=4,
        max_query_len=1,
        seq_lens=torch.tensor([8, 9, 0, 0], dtype=torch.int32),
        dcp_local_seq_lens=None,
    )

    captured = builder.build_for_cudagraph_capture(common)
    pointers = tuple(
        tensor.data_ptr()
        for tensor in (
            captured.selector_state_slot_ids,
            captured.selector_state_is_fresh,
            captured.selector_num_accepted_tokens,
            captured.selector_is_prefilling,
        )
    )
    assert torch.equal(
        captured.selector_state_slot_ids,
        torch.arange(4, dtype=torch.int32),
    )
    assert captured.selector_state_is_fresh.all()
    assert torch.equal(
        captured.selector_num_accepted_tokens,
        torch.ones(4, dtype=torch.int32),
    )
    assert not captured.selector_is_prefilling.any()

    runtime = builder.build(
        common_prefix_len=0,
        common_attn_metadata=common,
        selector_state_slot_ids=torch.tensor([7, 3, -1, -1], dtype=torch.int32),
        selector_state_is_fresh=torch.tensor([False, True, True, True]),
        selector_num_accepted_tokens=torch.tensor([4, 2, 1, 1], dtype=torch.int32),
        selector_is_prefilling=torch.tensor([False, True, False, False]),
    )
    assert (
        tuple(
            tensor.data_ptr()
            for tensor in (
                runtime.selector_state_slot_ids,
                runtime.selector_state_is_fresh,
                runtime.selector_num_accepted_tokens,
                runtime.selector_is_prefilling,
            )
        )
        == pointers
    )
    assert torch.equal(
        runtime.selector_state_slot_ids,
        torch.tensor([7, 3, -1, -1], dtype=torch.int32),
    )
    assert torch.equal(
        runtime.selector_state_is_fresh,
        torch.tensor([False, True, True, True]),
    )
    assert torch.equal(
        runtime.selector_num_accepted_tokens,
        torch.tensor([4, 2, 1, 1], dtype=torch.int32),
    )
    assert torch.equal(
        runtime.selector_is_prefilling,
        torch.tensor([False, True, False, False]),
    )


def test_b12x_sparse_mla_spec_decode_lengths_stay_in_builder_buffer(
    monkeypatch,
) -> None:
    """Multi-row decode lengths must live in the builder buffer.

    A FULL CUDA graph binds the tensor address at capture and replays against
    whatever a later build wrote there, so a fresh tensor per build leaves the
    replayed kernel reading stale lengths.
    """
    monkeypatch.setattr(
        SparseMLACommonMetadataBuilder,
        "build",
        lambda *args, **kwargs: SimpleNamespace(
            num_prefills=0,
            num_decodes=2,
            num_decode_tokens=8,
        ),
    )
    builder = B12xMLASparseMetadataBuilder.__new__(B12xMLASparseMetadataBuilder)
    builder.requires_glm_next_selector_metadata = False
    builder._ckv_gather_requested = False
    builder.dcp_world_size = 1
    builder._max_speculative_decode_query_len = 4
    builder.cache_seq_lens_per_token_buffer = torch.zeros(16, dtype=torch.int32)
    positions = torch.tensor([28, 29, 30, 31, 36, 37, 38, 39], dtype=torch.int64)
    common = SimpleNamespace(
        num_reqs=2,
        num_actual_tokens=8,
        max_query_len=4,
        seq_lens=torch.tensor([32, 40], dtype=torch.int32),
        dcp_local_seq_lens=None,
        positions=positions,
        is_prefilling=torch.zeros(2, dtype=torch.bool),
    )

    first = builder.build(common_prefix_len=0, common_attn_metadata=common)
    lengths = first.cache_seq_lens_per_token
    assert first.is_spec_decode
    assert lengths.data_ptr() == builder.cache_seq_lens_per_token_buffer.data_ptr()
    assert lengths.tolist() == [29, 30, 31, 32, 37, 38, 39, 40]

    common.positions = positions + 8
    second = builder.build(common_prefix_len=0, common_attn_metadata=common)
    assert second.cache_seq_lens_per_token.data_ptr() == lengths.data_ptr()
    assert lengths.tolist() == [37, 38, 39, 40, 45, 46, 47, 48]


def test_glm_selector_metadata_builder_requires_complete_runtime_state(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        SparseMLACommonMetadataBuilder,
        "build",
        lambda *args, **kwargs: SimpleNamespace(
            num_prefills=0,
            num_decode_tokens=0,
        ),
    )
    builder = _bare_glm_selector_metadata_builder()
    common = SimpleNamespace(
        num_reqs=1,
        num_actual_tokens=1,
        max_query_len=1,
        seq_lens=torch.ones(1, dtype=torch.int32),
        dcp_local_seq_lens=None,
    )

    with pytest.raises(RuntimeError, match="requires selector state slots"):
        builder.build(common_prefix_len=0, common_attn_metadata=common)


def test_glm_selector_metadata_builder_updates_draft_acceptance() -> None:
    builder = _bare_glm_selector_metadata_builder()
    accepted = torch.tensor([4, 2, 1, 1], dtype=torch.int32)
    metadata = SimpleNamespace(selector_num_accepted_tokens=accepted)

    builder.update_draft_decode_metadata(metadata)

    assert torch.equal(accepted, torch.ones(4, dtype=torch.int32))


def test_dsv4_metadata_builder_does_not_claim_glm_selector_state() -> None:
    builder = B12xMLASparseMetadataBuilder.__new__(B12xMLASparseMetadataBuilder)
    builder.requires_glm_next_selector_metadata = False

    assert builder._stage_glm_next_selector_metadata(
        num_reqs=2,
        for_cudagraph_capture=False,
        selector_state_slot_ids=None,
        selector_state_is_fresh=None,
        selector_num_accepted_tokens=None,
        selector_is_prefilling=None,
    ) == (None, None, None, None)
    with pytest.raises(TypeError, match="non-GLM"):
        builder._stage_glm_next_selector_metadata(
            num_reqs=2,
            for_cudagraph_capture=False,
            selector_state_slot_ids=torch.arange(2, dtype=torch.int32),
            selector_state_is_fresh=None,
            selector_num_accepted_tokens=None,
            selector_is_prefilling=None,
        )


def test_b12x_dsv4_backend_preserves_cache_contract() -> None:
    backend = b12x_mla.DeepseekV4B12xSparseMLABackend

    assert backend.get_name() == "B12X"
    assert "auto" in backend.supported_kv_cache_dtypes
    assert not backend.supports_pcp()
    assert not b12x_indexer.DeepseekV4B12xIndexerBackend.supports_pcp()

    storage = torch.empty((2, 600), dtype=torch.uint8)
    page_view = b12x_mla._cache_page_view(storage, page_size=1, name="cache")

    assert page_view.shape == (2, 584)
    assert page_view.stride() == (600, 1)
    assert (
        page_view.untyped_storage().data_ptr() == storage.untyped_storage().data_ptr()
    )


def test_b12x_non_compressed_indexer_exposes_scores_for_dcp(monkeypatch) -> None:
    calls: dict[str, Any] = {}

    def bind(bound_plan, **kwargs):
        calls["bind_plan"] = bound_plan
        calls["bind"] = kwargs
        return SimpleNamespace(
            output=kwargs["output_indices"],
            scores=kwargs["output_scores"],
        )

    plan = SimpleNamespace(
        shapes_and_dtypes=lambda: (((64,), torch.uint8),),
    )

    def run(binding):
        calls["run"] = binding
        binding.output.fill_(7)
        binding.scores.fill_(0.5)

    module = SimpleNamespace(
        Caps=lambda **kwargs: SimpleNamespace(**kwargs),
        PAGED_INDEX_PAGE_SIZE=64,
        plan=lambda caps: plan,
        bind=bind,
        run=run,
    )
    monkeypatch.setattr(generic_b12x_indexer, "_require_b12x_indexer", lambda: module)
    monkeypatch.setattr(
        generic_b12x_indexer,
        "current_workspace_manager",
        lambda: _Workspace(),
    )

    output = torch.empty((2, 4), dtype=torch.int32)
    scores = generic_b12x_indexer._run_paged_topk(
        module=module,
        plan=plan,
        q=torch.empty((2, 32, 128), dtype=torch.float8_e4m3fn),
        weights=torch.empty((2, 32), dtype=torch.float32),
        kv_cache=torch.empty((4, 64, 132), dtype=torch.uint8),
        seq_lens=torch.full((2,), 128, dtype=torch.int32),
        block_table=torch.zeros((2, 2), dtype=torch.int32),
        active_width=torch.full((1,), 128, dtype=torch.int32),
        output=output,
        return_scores=True,
    )

    assert calls["bind_plan"] is plan
    assert calls["bind"]["output_scores"] is scores
    assert calls["run"].scores is scores
    assert torch.count_nonzero(output != 7) == 0
    assert torch.count_nonzero(scores != 0.5) == 0


def test_b12x_compressed_sparse_mla_uses_public_plan_bind_run(
    monkeypatch,
) -> None:
    calls: dict[str, Any] = {}

    def make_caps(**kwargs):
        calls["caps"] = kwargs
        return SimpleNamespace(**kwargs)

    def bind(**kwargs):
        calls["bind"] = kwargs
        return SimpleNamespace(scratch=SimpleNamespace(mode=None))

    plan = SimpleNamespace(
        shapes_and_dtypes=lambda: (((32,), torch.uint8),),
        bind=bind,
    )

    def run(**kwargs):
        calls["run"] = kwargs
        kwargs["out"].fill_(3)

    module = SimpleNamespace(
        Caps=make_caps,
        plan=lambda caps: plan,
        run=run,
        split_chunks_for_contract=lambda **kwargs: 5,
    )
    monkeypatch.setattr(b12x_mla, "_require_b12x_compressed_sparse_mla", lambda: module)
    monkeypatch.setattr(b12x_mla, "current_workspace_manager", lambda: _Workspace())

    q = torch.empty((2, 16, 512), dtype=torch.bfloat16)
    output = torch.empty_like(q)
    b12x_mla._run_compressed_sparse_mla(
        q=q,
        output=output,
        attn_sink=torch.zeros((32,), dtype=torch.float32),
        scale=0.125,
        swa_k_cache=torch.empty((1, 584), dtype=torch.uint8),
        swa_indices=torch.zeros((2, 3), dtype=torch.int32),
        swa_lens=torch.full((2,), 3, dtype=torch.int32),
        swa_page_size=1,
        indexed_k_cache=torch.empty((1, 584), dtype=torch.uint8),
        indexed_indices=torch.zeros((2, 4), dtype=torch.int32),
        indexed_lens=torch.full((2,), 4, dtype=torch.int32),
        indexed_page_size=1,
        mode="decode",
        decode_row_capacity=8,
    )

    assert calls["caps"]["max_width"] == 7
    assert calls["caps"]["max_chunks_per_row"] == 5
    assert calls["bind"]["scratch"][0].dtype == torch.uint8
    assert calls["run"]["binding"].scratch.mode == "decode"
    assert calls["run"]["attn_sink"].shape == (16,)
    assert calls["run"]["out"] is output
    assert torch.count_nonzero(output != 3) == 0


def test_b12x_wo_projection_packs_and_runs_public_api(monkeypatch) -> None:
    calls: dict[str, Any] = {}

    def pack_weights(*args, **kwargs):
        calls["pack"] = (args, kwargs)
        return object()

    def run_inv_rope(*args, **kwargs):
        calls["run"] = (args, kwargs)
        return torch.full((args[0].shape[0], 256), 7, dtype=torch.bfloat16)

    module = SimpleNamespace(
        is_supported=lambda: True,
        pack_weights=pack_weights,
        run_inv_rope=run_inv_rope,
    )
    monkeypatch.setattr(b12x_mla, "get_b12x_wo_projection", lambda: module)
    monkeypatch.setattr(
        b12x_mla,
        "current_stream",
        lambda: SimpleNamespace(cuda_stream=123),
    )

    layer = object.__new__(b12x_mla.DeepseekV4B12xAttention)
    torch.nn.Module.__init__(layer)
    layer.n_local_groups = 2
    layer.n_local_heads = 4
    layer.head_dim = 128
    layer.nope_head_dim = 96
    layer.rope_head_dim = 32
    layer.o_lora_rank = 128
    layer.hidden_size = 256
    layer.rotary_emb = SimpleNamespace(cos_sin_cache=torch.empty((1, 64)))
    layer.wo_a = SimpleNamespace(
        weight=torch.empty((256, 256), dtype=torch.float8_e4m3fn),
        weight_scale_inv=torch.empty((2, 2), dtype=torch.float32),
        b12x_warmup_provider=object(),
    )
    layer.wo_b = SimpleNamespace(
        weight=torch.empty((256, 256), dtype=torch.float8_e4m3fn),
        weight_scale_inv=torch.empty((2, 2), dtype=torch.float32),
        b12x_warmup_provider=object(),
        reduce_results=False,
        tp_size=2,
    )
    layer._b12x_wo_projection_weights = None

    layer.setup_b12x_wo_projection()
    output = layer._o_proj(
        torch.empty((3, 4, 128), dtype=torch.bfloat16),
        torch.arange(3),
    )

    assert calls["pack"][1] == {
        "groups": 2,
        "group_width": 256,
        "rank": 128,
        "hidden": 256,
    }
    assert calls["run"][1]["heads_per_group"] == 2
    assert calls["run"][1]["stream"] == 123
    assert layer.wo_a.b12x_warmup_provider is None
    assert layer.wo_b.b12x_warmup_provider is None
    assert output.shape == (3, 256)
    assert torch.count_nonzero(output != 7) == 0


def test_b12x_mhc_uses_public_plan_bind_run(monkeypatch) -> None:
    calls: dict[str, Any] = {}
    retained_bindings: list[Any] = []

    def make_caps(**kwargs):
        calls["caps"] = kwargs
        return SimpleNamespace(**kwargs)

    def bind(plan, **kwargs):
        calls["bind"] = (plan, kwargs)
        return SimpleNamespace(**kwargs)

    plan = SimpleNamespace(
        shapes_and_dtypes=lambda: (((64,), torch.uint8),),
    )

    def run_pre(*args, **kwargs):
        calls["pre"] = (args, kwargs)
        binding = kwargs["binding"]
        return binding.out, binding.post, binding.comb, binding.y

    def run_post_pre(*args, **kwargs):
        calls["post_pre"] = (args, kwargs)
        binding = kwargs["binding"]
        return binding.out, binding.post, binding.comb, binding.y

    def run_post(*args):
        calls["post"] = args
        return args[1]

    module = SimpleNamespace(
        Caps=make_caps,
        DEFAULT_BLOCK_K=128,
        MULT=4,
        bind=bind,
        plan=lambda caps: plan,
        run_post=run_post,
        run_post_pre=run_post_pre,
        run_pre=run_pre,
    )
    monkeypatch.setattr(b12x_mla, "_require_b12x_mhc", lambda: module)
    monkeypatch.setattr(b12x_mla, "current_workspace_manager", lambda: _Workspace())
    monkeypatch.setattr(
        b12x_mla,
        "retain_cuda_graph_capture_resource",
        retained_bindings.append,
    )

    mhc = b12x_mla.B12xMHCResidual(
        hidden_size=256,
        hc_mult=4,
        rms_eps=1e-6,
        hc_eps=1e-6,
        sinkhorn_iters=20,
    )
    residual = torch.empty((3, 256), dtype=torch.bfloat16)
    hc_fn = torch.empty((24, 256), dtype=torch.float32)
    hc_scale = torch.empty((3,), dtype=torch.float32)
    hc_base = torch.empty((24,), dtype=torch.float32)
    norm_weight = torch.empty((256,), dtype=torch.bfloat16)

    residual_out, post, comb, layer_input = mhc.run_pre(
        residual,
        hc_fn,
        hc_scale,
        hc_base,
        norm_weight=norm_weight,
        norm_eps=1e-6,
    )
    next_outputs = mhc.run_post_pre(
        layer_input,
        residual_out,
        post,
        comb,
        torch.empty((24, 1024), dtype=torch.float32),
        hc_scale,
        hc_base,
        norm_weight=norm_weight,
        norm_eps=1e-6,
    )
    final = mhc.run_post(layer_input, *next_outputs[:3])

    assert calls["caps"]["hidden_size"] == 256
    assert calls["caps"]["split_k"] == 8
    assert calls["bind"][1]["scratch"].dtype == torch.uint8
    assert calls["pre"][1]["binding"].expected_m == 3
    assert calls["post_pre"][1]["expected_m"] == 3
    assert retained_bindings == [
        calls["pre"][1]["binding"],
        calls["post_pre"][1]["binding"],
    ]
    assert residual_out.shape == (3, 4, 256)
    assert layer_input.shape == (3, 256)
    assert final is next_outputs[0]


def test_b12x_dsa_indexer_uses_logical_slot_contract(monkeypatch) -> None:
    calls: dict[str, Any] = {}

    def make_caps(**kwargs):
        calls["caps"] = kwargs
        return SimpleNamespace(**kwargs)

    def bind(bound_plan, **kwargs):
        calls["bind_plan"] = bound_plan
        calls["bind"] = kwargs
        return SimpleNamespace(
            plan=bound_plan,
            route="packed_contiguous",
            output=kwargs["output_indices"],
        )

    plan = SimpleNamespace(
        layout=SimpleNamespace(route="packed_contiguous"),
        shapes_and_dtypes=lambda: (((64,), torch.uint8),),
    )

    def run(binding):
        calls["run"] = binding
        calls["output_before_run"] = binding.output.clone()
        binding.output.fill_(11)

    module = SimpleNamespace(
        Caps=make_caps,
        PAGED_INDEX_PAGE_SIZE=64,
        plan=lambda caps: plan,
        bind=bind,
        run=run,
    )
    monkeypatch.setattr(b12x_indexer, "_require_b12x_indexer", lambda: module)
    monkeypatch.setattr(b12x_indexer, "current_workspace_manager", lambda: _Workspace())

    output = torch.full((3, 4), 37, dtype=torch.int32)
    scores = torch.empty((3, 4), dtype=torch.float32)
    b12x_indexer._run_paged_topk(
        module=module,
        plan=plan,
        q=torch.empty((3, 16, 128), dtype=torch.float8_e4m3fn),
        weights=torch.empty((3, 16, 1), dtype=torch.float32),
        kv_cache=torch.empty((4, 64, 132), dtype=torch.uint8),
        seq_lens=torch.full((3,), 128, dtype=torch.int32),
        block_table=torch.zeros((3, 2), dtype=torch.int32),
        active_width=torch.full((1,), 128, dtype=torch.int32),
        output=output,
        scores=scores,
        shared_page_table=True,
    )

    assert calls["bind"]["output_scores"] is scores
    assert torch.count_nonzero(calls["output_before_run"] != 37) == 0

    builder = object.__new__(b12x_indexer.DeepseekV4B12xIndexerMetadataBuilder)
    builder.max_prefill_buffer_size = 1 << 30
    assert builder._supports_native_decode(8)
    assert builder._split_prefill_chunks(
        torch.tensor([64, 65536, 131072]),
        torch.tensor([1, 1]),
        num_decodes=1,
        max_logits_bytes=1 << 30,
    ) == [
        (slice(1, 2), slice(0, 1)),
        (slice(2, 3), slice(0, 1)),
    ]
    assert calls["bind_plan"] is plan
    assert calls["bind"]["active_width"].item() == 128
    assert calls["bind"]["output_indices"] is output
    assert calls["run"].output is output
    assert torch.count_nonzero(output != 11) == 0

    indexer = b12x_indexer.DeepseekV4B12xSparseIndexer(
        SimpleNamespace(),
        quant_block_size=128,
        scale_fmt="ue8m0",
        topk_tokens=512,
        head_dim=128,
        max_model_len=65536,
        max_total_seq_len=65536,
        topk_indices_buffer=torch.empty((2, 512), dtype=torch.int32),
        skip_k_cache_insert=True,
        compress_ratio=4,
    )
    indexer._reserve_profile_workspace(
        torch.empty((2, 64, 128), dtype=torch.float8_e4m3fn)
    )
    assert "source_layout" not in calls["caps"]
    assert "shared_page_table" not in calls["caps"]
    assert calls["caps"]["max_page_table_width"] == 1024


def test_b12x_dsa_indexer_reuses_plans_and_rebinds_shared_workspace(
    monkeypatch,
) -> None:
    calls = {"plan": 0, "workspace": 0, "bind": 0, "run": 0}

    def bind(bound_plan, **kwargs):
        calls["bind"] += 1
        return SimpleNamespace(
            plan=bound_plan,
            route="packed_contiguous",
            output=kwargs["output_indices"],
        )

    plan = SimpleNamespace(
        layout=SimpleNamespace(route="packed_contiguous"),
        shapes_and_dtypes=lambda: (((64,), torch.uint8),),
    )

    def make_plan(_caps):
        calls["plan"] += 1
        return plan

    def run(binding):
        calls["run"] += 1
        binding.output.fill_(7)

    module = SimpleNamespace(
        Caps=lambda **kwargs: SimpleNamespace(**kwargs),
        PAGED_INDEX_PAGE_SIZE=64,
        plan=make_plan,
        bind=bind,
        run=run,
    )

    class Workspace:
        def get_simultaneous(self, *shapes_and_dtypes):
            calls["workspace"] += 1
            return [
                torch.empty(shape, dtype=dtype) for shape, dtype in shapes_and_dtypes
            ]

    workspace = Workspace()
    monkeypatch.setattr(b12x_indexer, "_require_b12x_indexer", lambda: module)
    monkeypatch.setattr(
        b12x_indexer,
        "current_workspace_manager",
        lambda: workspace,
    )

    indexer = b12x_indexer.B12xC4SparseIndexer(
        SimpleNamespace(),
        quant_block_size=128,
        scale_fmt="ue8m0",
        topk_tokens=4,
        head_dim=128,
        max_model_len=128,
        max_total_seq_len=128,
        topk_indices_buffer=torch.empty((3, 4), dtype=torch.int32),
        skip_k_cache_insert=True,
        compress_ratio=4,
    )
    inputs = {
        "q": torch.empty((3, 16, 128), dtype=torch.float8_e4m3fn),
        "weights": torch.empty((3, 16, 1), dtype=torch.float32),
        "kv_cache": torch.empty((4, 64, 132), dtype=torch.uint8),
        "seq_lens": torch.full((3,), 128, dtype=torch.int32),
        "block_table": torch.zeros((3, 2), dtype=torch.int32),
        "output": torch.empty((3, 4), dtype=torch.int32),
        "shared_page_table": True,
    }

    indexer.run_paged_topk(**inputs)
    indexer.run_paged_topk(**inputs)

    assert calls == {"plan": 1, "workspace": 2, "bind": 2, "run": 2}
    assert torch.count_nonzero(inputs["output"] != 7) == 0


@pytest.mark.parametrize("rank", range(4))
@pytest.mark.parametrize("interleave", [1, 4])
@pytest.mark.parametrize("provided_local_lengths", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize(
    "rows_per_request,use_positions", [(1, False), (1, True), (4, True)]
)
def test_dcp_combine_uses_local_causal_lengths_in_stable_buffers(
    monkeypatch,
    rank,
    interleave,
    provided_local_lengths,
    rows_per_request,
    use_positions,
    device,
) -> None:
    """Draft rows can cross a shard boundary within one request."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    monkeypatch.setattr(
        SparseMLACommonMetadataBuilder,
        "build",
        lambda *args, **kwargs: SimpleNamespace(
            num_prefills=0, num_decodes=2, num_decode_tokens=2 * rows_per_request
        ),
    )
    builder = B12xMLASparseMetadataBuilder.__new__(B12xMLASparseMetadataBuilder)
    builder.requires_glm_next_selector_metadata = False
    builder._ckv_gather_requested = False
    builder.dcp_world_size = 4
    builder.dcp_rank = rank
    builder.cp_kv_cache_interleave_size = interleave
    builder._max_speculative_decode_query_len = 4
    builder.cache_seq_lens_per_token_buffer = torch.zeros(
        16, dtype=torch.int32, device=device
    )
    builder.dcp_combine_query_start_loc_buffer = torch.arange(
        17, dtype=torch.int32, device=device
    )

    def local_count(length):
        return sum((position // interleave) % 4 == rank for position in range(length))

    pointers = None
    for offset in (0, 8):
        positions = (
            torch.tensor(
                [start + row for start in (0, 2) for row in range(rows_per_request)],
                device=device,
            )
            + offset
        )
        global_lengths = [rows_per_request + offset, rows_per_request + 2 + offset]
        local_lengths = torch.tensor(
            [local_count(n) for n in global_lengths], dtype=torch.int32, device=device
        )
        common = SimpleNamespace(
            num_reqs=2,
            num_actual_tokens=2 * rows_per_request,
            max_query_len=rows_per_request,
            seq_lens=torch.tensor(global_lengths, dtype=torch.int32, device=device),
            dcp_local_seq_lens=local_lengths if provided_local_lengths else None,
            positions=positions if use_positions else None,
            is_prefilling=torch.zeros(2, dtype=torch.bool),
        )
        metadata = builder.build(0, common)
        torch.testing.assert_close(metadata.seq_lens, local_lengths)
        assert metadata.dcp_combine_seq_lens.tolist() == [
            local_count(int(p) + 1) for p in positions
        ]
        assert metadata.dcp_combine_query_start_loc.tolist() == list(
            range(2 * rows_per_request + 1)
        )
        assert metadata.dcp_combine_seq_lens is metadata.cache_seq_lens_per_token
        addresses = (
            metadata.dcp_combine_seq_lens.data_ptr(),
            metadata.dcp_combine_query_start_loc.data_ptr(),
        )
        if pointers is not None:
            assert addresses == pointers
        pointers = addresses
        if device == "cuda" and use_positions:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                replay_metadata = builder.build(0, common)
            positions.add_(1)
            expected = [local_count(int(p) + 1) for p in positions]
            for _ in range(3):
                graph.replay()
                assert replay_metadata.dcp_combine_seq_lens.tolist() == expected
                assert replay_metadata.dcp_combine_seq_lens.data_ptr() == pointers[0]
            assert not builder.cache_seq_lens_per_token_buffer[
                2 * rows_per_request :
            ].any()


@pytest.mark.parametrize("world_size", [1, 4])
def test_non_compressed_indexer_profiles_dcp_merge_capacity(monkeypatch, world_size):
    from vllm.v1.worker.workspace import WorkspaceManager, use_workspace_lane

    manager = WorkspaceManager(torch.device("cpu"), num_lanes=2)
    monkeypatch.setattr(
        generic_b12x_indexer, "current_workspace_manager", lambda: manager
    )
    indexer = object.__new__(generic_b12x_indexer.B12xSparseIndexer)
    torch.nn.Module.__init__(indexer)
    indexer.dcp_world_size = world_size
    indexer.topk_tokens = 2048
    indexer.topk_indices_buffer = torch.empty((8192, 2048), dtype=torch.int32)
    scratch_bytes = 96 * 1024**2
    plan = SimpleNamespace(
        caps=SimpleNamespace(max_q_rows=128),
        shapes_and_dtypes=lambda: (((scratch_bytes,), torch.uint8),),
    )
    indexer._decode_plans = {128: plan}
    indexer._prefill_plans = {128: plan}
    indexer._reserve_profile_workspace()
    manager.lock()
    if world_size == 1:
        assert manager._current_workspaces[1] is None
        assert manager._current_workspaces[0].numel() == scratch_bytes
        return
    for lane in range(2):
        with use_workspace_lane(lane):
            buffers = manager.get_simultaneous(
                *generic_b12x_indexer._dcp_merge_shapes(8192, 2048, world_size)
            )
            assert [buf.nbytes for buf in buffers] == [
                64 * 1024**2,
                128 * 1024**2,
                512 * 1024**2,
            ]
            base = buffers[0].data_ptr()
            assert buffers[1].data_ptr() == base + 64 * 1024**2
            assert buffers[2].data_ptr() == base + 192 * 1024**2
            assert manager._current_workspaces[lane].numel() == 704 * 1024**2
            # Indexer scratch and merge storage can alias only after scores.
            output = torch.empty((4, 2048), dtype=torch.int32)

            def run(binding):
                binding.scratch[0].fill_(255)
                binding.output_scores.fill_(1.5)

            scores = generic_b12x_indexer._run_paged_topk(
                module=SimpleNamespace(
                    bind=lambda plan, **kw: SimpleNamespace(**kw), run=run
                ),
                plan=plan,
                q=torch.empty((4, 1, 128)),
                weights=torch.empty((4, 1)),
                kv_cache=torch.empty((1, 64, 132), dtype=torch.uint8),
                seq_lens=torch.ones(4, dtype=torch.int32),
                block_table=torch.zeros((4, 1), dtype=torch.int32),
                active_width=torch.ones(1, dtype=torch.int32),
                output=output,
                return_scores=True,
            )
            score_view, packed, gathered = manager.get_simultaneous(
                *generic_b12x_indexer._dcp_merge_shapes(4, 2048, world_size)
            )
            assert scores.data_ptr() == score_view.data_ptr() == base
            packed.fill_(-2)
            gathered.fill_(-3)
            torch.testing.assert_close(scores, torch.full_like(scores, 1.5))


def test_dcp_candidate_gather_writes_rank_major_destination(monkeypatch):
    packed = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2)
    gathered = torch.empty((4, 2, 3, 2))
    addresses = []

    def all_gather(output, source):
        addresses.append(output.data_ptr())
        for rank in range(4):
            output[rank].copy_(source + rank)

    group = SimpleNamespace(
        device_communicator=SimpleNamespace(
            pynccl_comm=SimpleNamespace(disabled=False, all_gather=all_gather),
        )
    )
    generic_b12x_indexer._gather_dcp_candidates(group, packed, gathered)
    assert addresses == [gathered.data_ptr()]
    for rank in range(4):
        torch.testing.assert_close(gathered[rank], packed + rank)


@pytest.mark.parametrize("direct_dispatch", [False, True])
def test_dcp_merge_reuses_reserved_scores_and_gather_storage(
    monkeypatch, direct_dispatch
):
    pytest.importorskip("b12x.comm.pcie.dcp_candidate_topk")
    from vllm.v1.worker.workspace import WorkspaceManager

    rows, topk, ranks = 4, 2048, 4
    manager = WorkspaceManager(torch.device("cuda:0"))
    specs = generic_b12x_indexer._dcp_merge_shapes(rows, topk, ranks)
    manager.reserve_all(*specs)
    manager.lock()
    monkeypatch.setattr(
        generic_b12x_indexer, "current_workspace_manager", lambda: manager
    )
    scores, packed, gathered = manager.get_simultaneous(*specs)
    indices = torch.arange(topk, dtype=torch.int32, device="cuda").repeat(rows, 1)
    expected = indices.clone()
    scores.fill_(2)
    peers = torch.full_like(gathered, -1)
    peers[:, :, :, 0] = -torch.inf

    def all_gather(destination, source):
        destination.copy_(peers)
        destination[0].copy_(source)

    if direct_dispatch:
        from b12x.comm.pcie.dcp_candidate_topk import rank_major_topk

        from vllm.v1.attention.ops import b12x_dcp

        def merge_candidates(source, destination):
            assert source.data_ptr() == packed.data_ptr()
            assert destination.data_ptr() == indices.data_ptr()
            all_gather(gathered, source)
            rank_major_topk(gathered, destination)
            return True

        monkeypatch.setattr(
            b12x_dcp,
            "active_dcp_transport",
            lambda: SimpleNamespace(merge_candidates=merge_candidates),
        )
        monkeypatch.setattr(
            generic_b12x_indexer,
            "_gather_dcp_candidates",
            lambda *args: pytest.fail("Generic candidate exchange selected"),
        )

    monkeypatch.setattr(
        generic_b12x_indexer,
        "get_dcp_group",
        lambda: SimpleNamespace(
            device_communicator=SimpleNamespace(
                pynccl_comm=SimpleNamespace(
                    disabled=False,
                    all_gather=all_gather,
                )
            ),
        ),
    )
    generic_b12x_indexer._merge_dcp_topk(indices, scores, 0, ranks, 1)
    torch.testing.assert_close(indices.sort().values, expected * ranks)
    # Mutating outputs must not touch the scores or invalidate workspace views.
    torch.testing.assert_close(scores, torch.full_like(scores, 2))
    assert gathered.data_ptr() == manager.get_simultaneous(*specs)[2].data_ptr()
    indices.copy_(expected)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        generic_b12x_indexer._merge_dcp_topk(indices, scores, 0, ranks, 1)
    allocated = torch.accelerator.memory_allocated()
    for _ in range(3):
        indices.copy_(expected)
        graph.replay()
    torch.accelerator.synchronize()
    assert torch.accelerator.memory_allocated() == allocated
    torch.testing.assert_close(indices.sort().values, expected * ranks)
