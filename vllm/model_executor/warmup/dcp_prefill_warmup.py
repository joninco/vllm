# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Initialize optional DCP projections and collectives before graph capture."""

from vllm.platforms import current_platform


def warmup_dcp_prefill(worker, capture_sizes: list[int]) -> bool:
    """Initialize auxiliary DCP resources in the target and draft workspace lanes.

    Native attention plans are prepared by the startup coordinator. These
    hooks initialize projected prefill and candidate exchange before KV
    cache admission. Failed hooks remain eligible for a subsequent call.
    """
    if not current_platform.is_cuda():
        return False
    from vllm.model_executor.warmup.b12x_prepare import (
        _draft_lane,
        _draft_workload,
        _module_workload,
        b12x_workload,
    )
    from vllm.v1.worker.workspace import use_workspace_lane

    target = worker.get_model()
    draft = worker.get_draft_model()
    models = [model for model in (target, draft) if model is not None]
    if not any(
        callable(getattr(layer, "prepare_dcp_prefill", None))
        for model in models
        for layer in model.modules()
    ):
        return False
    workload = b12x_workload(worker, stage="state")
    workloads = [(target, workload)]
    if draft is not None:
        workloads.append(
            (draft, _draft_workload(worker, workload, lane=_draft_lane(worker)))
        )
    completed = getattr(worker, "_dcp_prefill_warmup_completed", None)
    if completed is None:
        completed = worker._dcp_prefill_warmup_completed = set()
    changed = False
    for model, model_workload in workloads:
        with use_workspace_lane(model_workload.lane):
            for layer in model.modules():
                prepare = getattr(layer, "prepare_dcp_prefill", None)
                if not callable(prepare):
                    continue
                counts = _module_workload(layer, model_workload).token_counts
                token_counts = tuple(sorted({*capture_sizes, *counts}))
                key = (id(layer), model_workload.lane, token_counts)
                if key not in completed:
                    changed = bool(prepare(token_counts)) or changed
                    completed.add(key)
    return changed
