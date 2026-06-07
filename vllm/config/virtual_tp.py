# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.attention.backends.registry import AttentionBackendEnum

if TYPE_CHECKING:
    from vllm.config.model import ModelConfig
    from vllm.config.parallel import ParallelConfig
    from vllm.config.vllm import VllmConfig

logger = init_logger(__name__)

VIRTUAL_TP_PLAN_ATTR = "vllm_virtual_tp_plan"
_VIRTUAL_TP_PLAN_KIND_B12X_PADDED = "b12x-padded"
_ATTENTION_HEAD_LOCAL_ALIGNMENT = 8
_MOE_INTERMEDIATE_LOCAL_ALIGNMENT = 32
_SHARED_EXPERT_FP8_LOCAL_ALIGNMENT = 128
_VOCAB_GLOBAL_ALIGNMENT = 64


def maybe_apply_b12x_virtual_tp_padding(vllm_config: VllmConfig) -> None:
    """Automatically pad config dimensions for B12X virtual TP sharding.

    Some B12X target models have dimensions that are not divisible by an
    otherwise useful TP size.  Native B12X kernels can run a larger logical
    per-rank shape as long as checkpoint tails are zero-filled during loading.
    This mutates the HuggingFace configs before vLLM's normal parallel-config
    verification and stores the original sizes in ``VIRTUAL_TP_PLAN_ATTR`` for
    weight loaders.
    """
    model_config = vllm_config.model_config
    if model_config is None:
        return

    plan_config = _get_plan_config(model_config)
    if getattr(plan_config, VIRTUAL_TP_PLAN_ATTR, None) is not None:
        return

    if not _is_supported_b12x_virtual_tp_config(model_config):
        return
    if not (_uses_b12x_attention(vllm_config) or _uses_native_b12x_moe(vllm_config)):
        return

    plan = _build_b12x_virtual_tp_plan(model_config, vllm_config.parallel_config)
    if not _plan_requires_padding(plan):
        return

    _validate_b12x_virtual_tp_config(vllm_config)

    _apply_b12x_virtual_tp_plan(model_config, plan)


def apply_b12x_virtual_tp_padding_to_model_config(
    model_config: ModelConfig,
    parallel_config: ParallelConfig,
) -> None:
    """Pad model dimensions when B12X virtual TP alignment requires it."""
    plan_config = _get_plan_config(model_config)
    if getattr(plan_config, VIRTUAL_TP_PLAN_ATTR, None) is not None:
        return

    if not _is_supported_b12x_virtual_tp_config(model_config):
        return

    plan = _build_b12x_virtual_tp_plan(model_config, parallel_config)
    if not _plan_requires_padding(plan):
        return

    _apply_b12x_virtual_tp_plan(model_config, plan)


def _build_b12x_virtual_tp_plan(
    model_config: ModelConfig,
    parallel_config: ParallelConfig,
) -> dict[str, dict[str, int] | str]:
    attention_tp_size = parallel_config.tensor_parallel_size
    moe_tp_size = (
        parallel_config.tensor_parallel_size
        * parallel_config.data_parallel_size
        * parallel_config.prefill_context_parallel_size
    )

    text_config = model_config.hf_text_config
    is_deepseek_v4 = _is_deepseek_v4_config(model_config)

    original_attention_heads = _require_int_attr(text_config, "num_attention_heads")
    attention_axis = _make_virtual_axis(
        original_attention_heads,
        attention_tp_size,
        _ATTENTION_HEAD_LOCAL_ALIGNMENT,
    )
    if is_deepseek_v4:
        original_output_groups = _require_int_attr(text_config, "o_groups")
        output_group_axis = _make_virtual_output_group_axis(
            original_output_groups,
            original_attention_heads,
            attention_axis["padded_size"],
            attention_tp_size,
        )
    else:
        output_group_axis = None

    moe_original_size = _require_int_attr(text_config, "moe_intermediate_size")
    moe_axis = _make_virtual_axis(
        moe_original_size,
        moe_tp_size,
        _MOE_INTERMEDIATE_LOCAL_ALIGNMENT,
    )
    shared_expert_axis = None
    n_shared_experts = getattr(text_config, "n_shared_experts", None)
    if is_deepseek_v4 and n_shared_experts is not None:
        shared_expert_axis = _make_virtual_axis(
            moe_original_size * int(n_shared_experts),
            attention_tp_size,
            _SHARED_EXPERT_FP8_LOCAL_ALIGNMENT,
        )
    vocab_axis = _make_virtual_vocab_axis(
        _require_int_attr(text_config, "vocab_size"),
        attention_tp_size,
    )

    plan: dict[str, dict[str, int] | str] = {
        "sharding": _VIRTUAL_TP_PLAN_KIND_B12X_PADDED,
        "attention_heads": attention_axis,
        "moe_intermediate_size": moe_axis,
        "vocab_size": vocab_axis,
    }
    if output_group_axis is not None:
        plan["output_groups"] = output_group_axis
    if shared_expert_axis is not None:
        plan["shared_expert_intermediate_size"] = shared_expert_axis
    return plan


def _plan_requires_padding(plan: dict[str, dict[str, int] | str]) -> bool:
    for axis in plan.values():
        if not isinstance(axis, dict):
            continue
        original_size = axis.get("original_size")
        padded_size = axis.get("padded_size")
        if (
            original_size is not None
            and padded_size is not None
            and int(original_size) != int(padded_size)
        ):
            return True
    return False


def _apply_b12x_virtual_tp_plan(
    model_config: ModelConfig,
    plan: dict[str, dict[str, int] | str],
) -> None:
    configs = tuple(_iter_virtual_tp_configs(model_config))
    attention_axis = _require_axis(plan, "attention_heads")
    moe_axis = _require_axis(plan, "moe_intermediate_size")
    vocab_axis = _require_axis(plan, "vocab_size")
    output_group_axis = _optional_axis(plan, "output_groups")
    shared_expert_axis = _optional_axis(plan, "shared_expert_intermediate_size")

    _set_all_config_attr(
        configs, "original_num_attention_heads", attention_axis["original_size"]
    )
    _set_existing_config_attr(
        configs, "num_attention_heads", attention_axis["padded_size"]
    )

    if output_group_axis is not None:
        _set_all_config_attr(
            configs, "original_o_groups", output_group_axis["original_size"]
        )
        _set_existing_config_attr(configs, "o_groups", output_group_axis["padded_size"])
    _set_all_config_attr(
        configs, "original_moe_intermediate_size", moe_axis["original_size"]
    )
    _set_existing_config_attr(configs, "moe_intermediate_size", moe_axis["padded_size"])

    for config in configs:
        setattr(config, VIRTUAL_TP_PLAN_ATTR, plan)

    model_config.model_arch_config = model_config.get_model_arch_config()

    if output_group_axis is None:
        logger.warning(
            "Automatically enabled B12X virtual TP padding for B12X kernel "
            "compatibility: attention heads %d -> %d, MoE intermediate "
            "size %d -> %d, vocab size %d -> %d.",
            attention_axis["original_size"],
            attention_axis["padded_size"],
            moe_axis["original_size"],
            moe_axis["padded_size"],
            vocab_axis["original_size"],
            vocab_axis["padded_size"],
        )
    else:
        logger.warning(
            "Automatically enabled B12X virtual TP padding for B12X kernel "
            "compatibility: attention heads %d -> %d, output groups %d -> %d, "
            "MoE intermediate size %d -> %d, vocab size %d -> %d.",
            attention_axis["original_size"],
            attention_axis["padded_size"],
            output_group_axis["original_size"],
            output_group_axis["padded_size"],
            moe_axis["original_size"],
            moe_axis["padded_size"],
            vocab_axis["original_size"],
            vocab_axis["padded_size"],
        )
    if shared_expert_axis is not None:
        logger.warning(
            "Automatically enabled B12X virtual TP padding for shared experts: "
            "intermediate size %d -> %d.",
            shared_expert_axis["original_size"],
            shared_expert_axis["padded_size"],
        )


def _require_axis(plan: dict[str, dict[str, int] | str], name: str) -> dict[str, int]:
    axis = plan.get(name)
    if not isinstance(axis, dict):
        raise ValueError(f"B12X virtual TP plan missing axis {name!r}.")
    return axis


def _optional_axis(
    plan: dict[str, dict[str, int] | str], name: str
) -> dict[str, int] | None:
    axis = plan.get(name)
    if axis is None:
        return None
    if not isinstance(axis, dict):
        raise ValueError(f"B12X virtual TP plan axis {name!r} is invalid.")
    return axis


def _validate_b12x_virtual_tp_config(vllm_config: VllmConfig) -> None:
    parallel_config = vllm_config.parallel_config
    model_config = vllm_config.model_config
    assert model_config is not None

    if not _is_supported_b12x_virtual_tp_config(model_config):
        raise ValueError(
            "B12X virtual TP padding is currently supported only for "
            "DeepSeek V4 and sparse MLA/DSA models."
        )

    if parallel_config.enable_expert_parallel:
        raise ValueError(
            "B12X virtual TP padding is incompatible with expert "
            "parallelism. Use tensor parallelism for the B12X padded path."
        )

    if vllm_config.kernel_config.moe_backend == "deep_gemm_mega_moe":
        raise ValueError(
            "B12X virtual TP padding is incompatible with DeepGEMM MegaMoE."
        )

    if not _uses_native_b12x_moe(vllm_config):
        raise ValueError(
            "B12X virtual TP padding requires the native B12X MoE "
            "backend. Pass --moe-backend b12x or set VLLM_USE_B12X_MOE=1."
        )

    if not _uses_b12x_attention(vllm_config):
        raise ValueError(
            "B12X virtual TP padding requires the B12X MLA sparse "
            "attention backend."
        )


def _is_deepseek_v4_config(model_config: ModelConfig) -> bool:
    for config in _iter_virtual_tp_configs(model_config):
        if getattr(config, "model_type", None) == "deepseek_v4":
            return True
        architectures = getattr(config, "architectures", None) or ()
        if (
            "DeepseekV4ForCausalLM" in architectures
            or "DeepseekV4ForCausalLMNextN" in architectures
        ):
            return True

    text_config = model_config.hf_text_config
    return (
        hasattr(text_config, "o_groups")
        and hasattr(text_config, "moe_intermediate_size")
        and hasattr(text_config, "n_routed_experts")
    )


def _is_sparse_mla_config(model_config: ModelConfig) -> bool:
    for config in _iter_virtual_tp_configs(model_config):
        if (
            getattr(config, "kv_lora_rank", None) is not None
            and getattr(config, "qk_rope_head_dim", None) is not None
            and _positive_int_attr(config, "index_topk")
        ):
            return True
    return False


def _is_supported_b12x_virtual_tp_config(model_config: ModelConfig) -> bool:
    return _is_deepseek_v4_config(model_config) or _is_sparse_mla_config(model_config)


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
        and _is_supported_b12x_virtual_tp_config(model_config)
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


def _make_virtual_vocab_axis(
    original_size: int,
    tp_size: int,
) -> dict[str, int]:
    padding_size = math.lcm(_VOCAB_GLOBAL_ALIGNMENT, tp_size)
    padded_size = math.ceil(original_size / padding_size) * padding_size
    assert padded_size % tp_size == 0
    return {
        "original_size": original_size,
        "padded_size": padded_size,
        "tp_size": tp_size,
        "local_size": padded_size // tp_size,
        "padding_size": padding_size,
    }


def _make_virtual_output_group_axis(
    original_size: int,
    original_attention_heads: int,
    padded_attention_heads: int,
    tp_size: int,
) -> dict[str, int]:
    if original_attention_heads % original_size != 0:
        raise ValueError(
            "DeepSeek V4 virtual TP padding requires num_attention_heads to "
            "be divisible by o_groups."
        )

    heads_per_group = original_attention_heads // original_size
    if padded_attention_heads % heads_per_group != 0:
        raise ValueError(
            "DeepSeek V4 virtual TP padding produced attention heads that do "
            "not preserve the original heads-per-output-group ratio."
        )

    padded_size = padded_attention_heads // heads_per_group
    if padded_size % tp_size != 0:
        raise ValueError(
            "DeepSeek V4 virtual TP padding produced output groups that are "
            "not divisible by tensor parallel size."
        )

    return {
        "original_size": original_size,
        "padded_size": padded_size,
        "tp_size": tp_size,
        "local_size": padded_size // tp_size,
        "heads_per_group": heads_per_group,
    }


def _require_int_attr(config: Any, attr: str) -> int:
    value = getattr(config, attr, None)
    if value is None:
        raise ValueError(
            f"B12X virtual TP padding requires config attribute {attr!r}."
        )
    return int(value)


def _positive_int_attr(config: Any, attr: str) -> bool:
    value = getattr(config, attr, None)
    if value is None:
        return False
    try:
        return int(value) > 0
    except (TypeError, ValueError):
        return False


def _get_plan_config(model_config: ModelConfig) -> Any:
    return model_config.hf_text_config or model_config.hf_config


def _iter_virtual_tp_configs(model_config: ModelConfig) -> Iterable[Any]:
    hf_config = model_config.hf_config
    yield from _unique_configs(
        (
            hf_config,
            model_config.hf_text_config,
            getattr(hf_config, "text_config", None),
        )
    )


def _unique_configs(configs: Iterable[Any]) -> Iterable[Any]:
    seen: set[int] = set()
    for config in configs:
        if config is None:
            continue
        config_id = id(config)
        if config_id in seen:
            continue
        seen.add(config_id)
        yield config


def _set_existing_config_attr(configs: Iterable[Any], attr: str, value: int) -> None:
    for config in configs:
        if hasattr(config, attr):
            setattr(config, attr, value)


def _set_all_config_attr(configs: Iterable[Any], attr: str, value: int) -> None:
    for config in configs:
        setattr(config, attr, value)
