# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Literal

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.attention.backends.registry import AttentionBackendEnum

if TYPE_CHECKING:
    from vllm.config.model import ModelConfig
    from vllm.config.vllm import VllmConfig

logger = init_logger(__name__)

VirtualTPSharding = Literal["off", "b12x-padded"]

VIRTUAL_TP_SHARDING_OFF: VirtualTPSharding = "off"
VIRTUAL_TP_SHARDING_B12X_PADDED: VirtualTPSharding = "b12x-padded"
VIRTUAL_TP_PLAN_ATTR = "vllm_virtual_tp_plan"
_SHARED_EXPERT_FP8_LOCAL_ALIGNMENT = 128


def maybe_apply_b12x_virtual_tp_padding(vllm_config: VllmConfig) -> None:
    """Pad DeepSeek V4 config dimensions for B12X virtual TP sharding.

    Some DeepSeek V4 Pro dimensions are not divisible by TP=10.  Native B12X
    kernels can run a larger logical per-rank shape as long as checkpoint tails
    are zero-filled during loading.  This mutates the HuggingFace configs before
    vLLM's normal parallel-config verification and stores the original sizes in
    ``VIRTUAL_TP_PLAN_ATTR`` for weight loaders.
    """
    parallel_config = vllm_config.parallel_config
    if parallel_config.virtual_tp_sharding == VIRTUAL_TP_SHARDING_OFF:
        return

    if parallel_config.virtual_tp_sharding != VIRTUAL_TP_SHARDING_B12X_PADDED:
        raise ValueError(
            "Unsupported virtual TP sharding mode "
            f"{parallel_config.virtual_tp_sharding!r}."
        )

    model_config = vllm_config.model_config
    if model_config is None:
        return

    text_config = model_config.hf_text_config
    if getattr(text_config, VIRTUAL_TP_PLAN_ATTR, None) is not None:
        return

    _validate_b12x_virtual_tp_config(vllm_config)

    attention_tp_size = parallel_config.tensor_parallel_size
    moe_tp_size = (
        parallel_config.tensor_parallel_size
        * parallel_config.data_parallel_size
        * parallel_config.prefill_context_parallel_size
    )

    attention_axis = _make_virtual_axis(
        _require_int_attr(text_config, "num_attention_heads"),
        attention_tp_size,
        parallel_config.b12x_virtual_tp_attention_head_alignment,
    )
    output_group_axis = _make_virtual_axis(
        _require_int_attr(text_config, "o_groups"),
        attention_tp_size,
    )
    moe_original_size = _require_int_attr(text_config, "moe_intermediate_size")
    moe_axis = _make_virtual_axis(
        moe_original_size,
        moe_tp_size,
        parallel_config.b12x_virtual_tp_moe_intermediate_alignment,
    )
    shared_expert_axis = None
    n_shared_experts = getattr(text_config, "n_shared_experts", None)
    if n_shared_experts is not None:
        shared_expert_axis = _make_virtual_axis(
            moe_original_size * int(n_shared_experts),
            attention_tp_size,
            _SHARED_EXPERT_FP8_LOCAL_ALIGNMENT,
        )

    configs = tuple(
        _unique_configs((model_config.hf_config, model_config.hf_text_config))
    )
    _set_existing_config_attr(
        configs, "num_attention_heads", attention_axis["padded_size"]
    )
    _set_existing_config_attr(configs, "o_groups", output_group_axis["padded_size"])
    _set_existing_config_attr(configs, "moe_intermediate_size", moe_axis["padded_size"])

    plan = {
        "sharding": VIRTUAL_TP_SHARDING_B12X_PADDED,
        "attention_heads": attention_axis,
        "output_groups": output_group_axis,
        "moe_intermediate_size": moe_axis,
    }
    if shared_expert_axis is not None:
        plan["shared_expert_intermediate_size"] = shared_expert_axis

    for config in configs:
        setattr(config, VIRTUAL_TP_PLAN_ATTR, plan)

    model_config.model_arch_config = model_config.get_model_arch_config()

    logger.info(
        "Enabled B12X virtual TP padding: attention heads %d -> %d, "
        "output groups %d -> %d, MoE intermediate size %d -> %d.",
        attention_axis["original_size"],
        attention_axis["padded_size"],
        output_group_axis["original_size"],
        output_group_axis["padded_size"],
        moe_axis["original_size"],
        moe_axis["padded_size"],
    )
    if shared_expert_axis is not None:
        logger.info(
            "Enabled B12X virtual TP padding for shared experts: "
            "intermediate size %d -> %d.",
            shared_expert_axis["original_size"],
            shared_expert_axis["padded_size"],
        )


def _validate_b12x_virtual_tp_config(vllm_config: VllmConfig) -> None:
    parallel_config = vllm_config.parallel_config
    model_config = vllm_config.model_config
    assert model_config is not None

    if not _is_deepseek_v4_config(model_config):
        raise ValueError(
            "--virtual-tp-sharding=b12x-padded is currently supported only "
            "for DeepSeek V4 models."
        )

    if parallel_config.enable_expert_parallel:
        raise ValueError(
            "--virtual-tp-sharding=b12x-padded is incompatible with expert "
            "parallelism. Use tensor parallelism for the B12X padded path."
        )

    if vllm_config.kernel_config.moe_backend == "deep_gemm_mega_moe":
        raise ValueError(
            "--virtual-tp-sharding=b12x-padded is incompatible with DeepGEMM MegaMoE."
        )

    if not _uses_native_b12x_moe(vllm_config):
        raise ValueError(
            "--virtual-tp-sharding=b12x-padded requires the native B12X MoE "
            "backend. Pass --moe-backend b12x or set VLLM_USE_B12X_MOE=1."
        )

    if not _uses_b12x_attention(vllm_config):
        raise ValueError(
            "--virtual-tp-sharding=b12x-padded requires the B12X MLA sparse "
            "attention backend."
        )

    if parallel_config.b12x_virtual_tp_attention_head_alignment <= 0:
        raise ValueError("b12x_virtual_tp_attention_head_alignment must be > 0.")
    if parallel_config.b12x_virtual_tp_moe_intermediate_alignment <= 0:
        raise ValueError("b12x_virtual_tp_moe_intermediate_alignment must be > 0.")


def _is_deepseek_v4_config(model_config: ModelConfig) -> bool:
    hf_config = model_config.hf_config
    text_config = model_config.hf_text_config
    return (
        getattr(hf_config, "model_type", None) == "deepseek_v4"
        or getattr(text_config, "model_type", None) == "deepseek_v4"
        or (
            hasattr(text_config, "o_groups")
            and hasattr(text_config, "moe_intermediate_size")
            and hasattr(text_config, "n_routed_experts")
        )
    )


def _uses_native_b12x_moe(vllm_config: VllmConfig) -> bool:
    moe_backend = vllm_config.kernel_config.moe_backend
    return moe_backend == "b12x" or (moe_backend == "auto" and envs.VLLM_USE_B12X_MOE)


def _uses_b12x_attention(vllm_config: VllmConfig) -> bool:
    backend = getattr(vllm_config.attention_config, "backend", None)
    if backend == AttentionBackendEnum.B12X_MLA_SPARSE:
        return True

    model_config = vllm_config.model_config
    return (
        model_config is not None
        and _is_deepseek_v4_config(model_config)
        and current_platform.is_cuda()
        and current_platform.has_device_capability(120)
    )


def _make_virtual_axis(
    original_size: int,
    tp_size: int,
    local_alignment: int = 1,
) -> dict[str, int]:
    local_size = math.ceil(original_size / tp_size)
    local_size = math.ceil(local_size / local_alignment) * local_alignment
    return {
        "original_size": original_size,
        "padded_size": local_size * tp_size,
        "tp_size": tp_size,
        "local_size": local_size,
    }


def _require_int_attr(config: Any, attr: str) -> int:
    value = getattr(config, attr, None)
    if value is None:
        raise ValueError(
            "--virtual-tp-sharding=b12x-padded requires DeepSeek V4 config "
            f"attribute {attr!r}."
        )
    return int(value)


def _unique_configs(configs: Iterable[Any]) -> Iterable[Any]:
    seen: set[int] = set()
    for config in configs:
        config_id = id(config)
        if config_id in seen:
            continue
        seen.add(config_id)
        yield config


def _set_existing_config_attr(configs: Iterable[Any], attr: str, value: int) -> None:
    for config in configs:
        if hasattr(config, attr):
            setattr(config, attr, value)
