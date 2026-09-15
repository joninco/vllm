# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from torch import nn

from vllm.utils.mem_constants import GiB_bytes
from vllm.v1.worker import gpu_worker, startup_plan
from vllm.v1.worker.startup_plan import (
    maybe_apply_startup_plan,
    maybe_save_startup_plan,
)


def test_mark_b12x_eager_shapes_covers_encoder_and_connector_profile_shapes(
    monkeypatch,
) -> None:
    import vllm.multimodal.encoder_budget as encoder_budget
    from vllm.model_executor.warmup.b12x_prepare import mark_b12x_eager_shapes

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.visual = nn.Module()
            self.visual.block = nn.Module()
            self.visual.merger = nn.Sequential(nn.Module())
            self.visual.deepstack_merger_list = nn.ModuleList([nn.Module()])

        def get_num_mm_encoder_tokens(self, tokens):
            return tokens * 4

        def get_num_mm_connector_tokens(self, tokens):
            return tokens // 4

        def get_mm_mapping(self):
            return SimpleNamespace(
                connector=("visual.merger", "visual.deepstack_merger_list")
            )

    class _FakeBudget:
        def __init__(self, vllm_config, mm_registry, enable_cache):
            del vllm_config, mm_registry, enable_cache

        def get_encoder_budget(self):
            return 16_384

    monkeypatch.setattr(encoder_budget, "MultiModalBudget", _FakeBudget)
    model = _Model()
    worker = SimpleNamespace(
        get_model=lambda: model,
        vllm_config=SimpleNamespace(),
        model_runner=SimpleNamespace(mm_registry=object()),
    )

    mark_b12x_eager_shapes(worker)

    assert model.visual.block.b12x_eager_token_counts == (65_536,)
    assert model.visual.block.b12x_eager_only is True
    for connector in (model.visual.merger, model.visual.deepstack_merger_list):
        assert connector.b12x_eager_token_counts == (16_384,)
        assert all(module.b12x_eager_only for module in connector.modules())


def test_b12x_workload_covers_target_and_draft_profile_shapes() -> None:
    from vllm.model_executor.warmup.b12x_prepare import b12x_workload

    compilation = SimpleNamespace(
        cudagraph_capture_sizes=(1, 2, 4, 8),
        compile_sizes=(),
        get_compile_ranges=lambda: (SimpleNamespace(end=128),),
    )
    worker = SimpleNamespace(
        get_model=lambda: nn.Module(),
        model_runner=SimpleNamespace(mm_registry=None),
        vllm_config=SimpleNamespace(
            compilation_config=compilation,
            speculative_config=SimpleNamespace(num_speculative_tokens=3),
        ),
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=128,
            max_num_seqs=1,
        ),
        model_config=SimpleNamespace(dtype="bf16", max_model_len=4096),
    )

    workload = b12x_workload(worker, stage="weights")

    # b12x_preparation_token_counts also reserves the post-speculative decode
    # regime (max_tokens - speculative_tokens = 128 - 3 = 125).
    assert workload.token_counts == (1, 2, 4, 8, 125, 128)
    assert workload.fixed_token_counts == (1, 2, 4, 8)
    assert workload.max_tokens == 128
    assert workload.speculative_tokens == 3


# Startup-plan persistence (vllm/v1/worker/startup_plan.py), applied and
# saved by Worker.determine_available_memory / compile_or_warm_up_model.


def _plan_worker(config_hash="abc123", free_memory=78 * GiB_bytes, kv_bytes=None):
    """The minimal Worker surface the startup-plan entry points touch."""
    return SimpleNamespace(
        vllm_config=SimpleNamespace(compute_hash=lambda: config_hash),
        rank=0,
        parallel_config=SimpleNamespace(world_size=1),
        init_snapshot=SimpleNamespace(free_memory=free_memory),
        cache_config=SimpleNamespace(kv_cache_memory_bytes=kv_bytes),
    )


def _plan_platform(name="NVIDIA H100 PCIe"):
    return SimpleNamespace(
        get_device_name=lambda device_id=0: name,
        get_device_total_memory=lambda device_id=0: 80 * GiB_bytes,
        get_device_capability=lambda device_id=0: (9, 0),
    )


@pytest.fixture
def plan_env(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """Enable the startup plan, isolated under a tmp cache root."""
    monkeypatch.setenv("VLLM_ENABLE_STARTUP_PLAN", "1")
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path))
    with patch.object(startup_plan, "current_platform", _plan_platform()):
        yield


def test_startup_plan_fingerprint_sensitivity(plan_env):
    """The fingerprint is the OOM-safety key: stable for identical inputs,
    different for anything the profiled value depends on."""
    fp = startup_plan.compute_plan_fingerprint
    base = fp(_plan_worker().vllm_config, 0, 1)
    assert base == fp(_plan_worker().vllm_config, 0, 1)
    assert base != fp(_plan_worker("other").vllm_config, 0, 1)
    assert base != fp(_plan_worker().vllm_config, 1, 2)
    with patch.object(startup_plan, "current_platform", _plan_platform("NVIDIA A100")):
        assert base != fp(_plan_worker().vllm_config, 0, 1)
    with patch("vllm.__version__", "0.0.0+plan-test"):
        assert base != fp(_plan_worker().vllm_config, 0, 1)


def test_startup_plan_apply_gate(plan_env):
    """Only a fingerprint-matching, memory-safe plan is ever applied."""
    maybe_save_startup_plan(_plan_worker(), 50 * GiB_bytes)

    applied = _plan_worker()
    maybe_apply_startup_plan(applied)
    assert applied.cache_config.kv_cache_memory_bytes == 50 * GiB_bytes

    less_memory = _plan_worker(free_memory=60 * GiB_bytes)
    other_config = _plan_worker(config_hash="zzz999")
    for refused in (less_memory, other_config):
        maybe_apply_startup_plan(refused)
        assert refused.cache_config.kv_cache_memory_bytes is None

    # An explicit --kv-cache-memory is never overridden.
    explicit = _plan_worker(kv_bytes=7 * GiB_bytes)
    maybe_apply_startup_plan(explicit)
    assert explicit.cache_config.kv_cache_memory_bytes == 7 * GiB_bytes


@pytest.mark.parametrize(
    "final_free_memory,cudagraph_estimate,expected_available_memory",
    [
        # No late persistent memory, no graph estimate.
        (80, 0, 78),
        # Five bytes retained after the graph profile, charged in full.
        (75, 0, 73),
        # The graph estimate already covers three of the five retained bytes.
        (75, 3, 73),
        # The graph estimate covers every retained byte; only it is charged.
        (75, 6, 72),
    ],
)
@pytest.mark.parametrize("estimate_graphs", [False, True])
@pytest.mark.parametrize("resolves_kernels", [True, False])
@pytest.mark.parametrize("initial_device_charge", [0, 3])
@pytest.mark.parametrize(
    "execution_mode", ["graphs", "eager_projected", "eager_disabled"]
)
def test_kv_memory_profile_uses_repeatable_peak_before_cudagraphs(
    monkeypatch,
    final_free_memory,
    cudagraph_estimate,
    expected_available_memory,
    estimate_graphs,
    resolves_kernels,
    initial_device_charge,
    execution_mode,
):
    """KV sizing must retain the warmed allocator and graph high-waters and
    the persistent allocations made after the activation profile, charging
    the part of them inside the CUDA-graph estimate once. The expected
    budgets of the table apply when the estimate is applied; without it the
    retained memory is charged in full and nothing is subtracted for graphs.
    The repeated profile runs only when the B12X warm-up resolved kernels;
    otherwise the single profile's headroom stands and no correction is
    subtracted."""
    events: list[object] = []
    profile_graphs = execution_mode == "graphs"
    needs_warmup = execution_mode != "eager_disabled"
    did_warmup = needs_warmup and resolves_kernels
    first_profile = SimpleNamespace(
        free_memory=84,
        torch_allocated=7,
        torch_memory=8,
        non_torch_memory=2,
    )
    after_warmup = SimpleNamespace(
        free_memory=83,
        torch_allocated=8,
        torch_memory=9,
        non_torch_memory=2,
    )
    repeatable_profile = SimpleNamespace(
        free_memory=82,
        torch_allocated=9,
        torch_memory=16,
        non_torch_memory=2,
    )
    final = SimpleNamespace(
        free_memory=final_free_memory,
        torch_allocated=8,
        torch_memory=9,
        non_torch_memory=3,
    )
    snapshots = iter(
        [first_profile, after_warmup, repeatable_profile, final]
        if did_warmup
        else [first_profile, final]
    )

    def profile_cudagraph_memory(prepare_profile_state):
        prepare_profile_state()
        events.append("profile_cudagraph_memory")
        return cudagraph_estimate

    def reserve_sampler_workspace():
        events.append("reserve_sampler_workspace")
        return 0

    configured_layer = SimpleNamespace(
        dcp_manager=SimpleNamespace(
            prefill_warmup_key=("projected",)
            if execution_mode == "eager_projected"
            else None
        )
    )

    def run_profile(name, prepare):
        prepare()
        events.append(name)

    model_runner = SimpleNamespace(
        get_model=lambda: SimpleNamespace(modules=lambda: [configured_layer]),
        model_memory_usage=0,
        reserve_sampler_workspace=reserve_sampler_workspace,
        profile_run=lambda prepare: run_profile("profile_run", prepare),
        profile_glm_dcp_attention=lambda prepare: run_profile(
            "profile_glm_dcp_attention", prepare
        ),
        profile_cudagraph_memory=profile_cudagraph_memory,
    )
    profile_result = SimpleNamespace(
        weights_memory=0,
        total_consumed=10,
        transient_peak_headroom=5,
        before_profile=SimpleNamespace(free_memory=85),
        after_profile=SimpleNamespace(
            free_memory=80,
            torch_allocated=8,
            torch_memory=9,
            non_torch_memory=3,
        ),
        non_kv_cache_memory=10,
    )

    @contextmanager
    def fake_memory_profiling(*args, **kwargs):
        yield profile_result

    worker = SimpleNamespace(
        cache_config=SimpleNamespace(
            kv_cache_memory_bytes=None,
            gpu_memory_utilization=0.9,
        ),
        model_runner=model_runner,
        init_snapshot=SimpleNamespace(
            free_memory=100,
            total_memory=100,
            torch_allocated=1,
            torch_memory=1,
            non_torch_memory=1,
        ),
        requested_memory=90,
        initial_device_memory_charge=initial_device_charge,
        device="cuda:0",
        model_config=SimpleNamespace(multimodal_config=None),
        parallel_config=SimpleNamespace(decode_context_parallel_size=4),
        _prepare_b12x_profile_state=lambda: events.append("prepare_b12x_profile_state"),
        _release_b12x_profile_state=lambda: events.append("release_b12x_profile_state"),
        vllm_config=SimpleNamespace(
            compilation_config=SimpleNamespace(
                cudagraph_mode=(
                    gpu_worker.CUDAGraphMode.PIECEWISE
                    if profile_graphs
                    else gpu_worker.CUDAGraphMode.NONE
                ),
                cudagraph_capture_sizes=[8, 4],
            )
        ),
    )

    monkeypatch.setattr(gpu_worker, "maybe_apply_startup_plan", lambda worker: None)
    monkeypatch.setenv(
        "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS", str(int(estimate_graphs))
    )
    monkeypatch.setattr(gpu_worker, "memory_profiling", fake_memory_profiling)
    monkeypatch.setattr(
        gpu_worker,
        "MemorySnapshot",
        lambda *, device: next(snapshots),
    )
    monkeypatch.setattr(
        gpu_worker,
        "current_platform",
        SimpleNamespace(is_cuda_alike=lambda: True),
    )

    def fake_b12x_warmup(worker, sizes):
        events.append(("b12x_warmup", tuple(sizes)))
        return resolves_kernels

    monkeypatch.setattr(gpu_worker, "warmup_dcp_prefill", fake_b12x_warmup)
    monkeypatch.setattr(
        gpu_worker.torch.accelerator,
        "reset_peak_memory_stats",
        lambda device: events.append(("reset_peak", device)),
    )
    monkeypatch.setattr(
        gpu_worker.torch.accelerator,
        "empty_cache",
        lambda: events.append("empty_cache"),
    )
    monkeypatch.setattr(
        gpu_worker,
        "reserve_mm_ipc_gpu_memory",
        lambda requested, *args: requested,
    )

    available = gpu_worker.Worker.determine_available_memory(worker)

    # The cached allocator blocks are released before the final snapshot, as
    # the activation profile released them before its own after-profile
    # snapshot, so only allocations retained after the CUDA-graph profile
    # count as late persistent memory.
    repeated_profile: list[object] = (
        [
            ("reset_peak", "cuda:0"),
            "prepare_b12x_profile_state",
            "profile_run",
            "release_b12x_profile_state",
            "prepare_b12x_profile_state",
            "profile_glm_dcp_attention",
            "release_b12x_profile_state",
        ]
        if did_warmup
        else []
    )
    assert events == [
        "reserve_sampler_workspace",
        "prepare_b12x_profile_state",
        "profile_run",
        "release_b12x_profile_state",
        "prepare_b12x_profile_state",
        "profile_glm_dcp_attention",
        "release_b12x_profile_state",
        *([("b12x_warmup", (8, 4))] if needs_warmup else []),
        *repeated_profile,
        *(
            [
                "prepare_b12x_profile_state",
                "profile_cudagraph_memory",
                "release_b12x_profile_state",
            ]
            if profile_graphs
            else []
        ),
        "empty_cache",
    ]
    # The repeatable profile retained seven bytes above its cleanup state.
    # Five are already covered by the live-allocation peak, leaving two bytes
    # of allocator-reservation headroom to deduct from KV capacity, plus the
    # free memory retained after the activation profile. Without the repeat
    # those two bytes are not deducted.
    headroom_correction = 2 if did_warmup else 0
    late_persistent_memory = 80 - final_free_memory
    if estimate_graphs and profile_graphs:
        assert (
            available
            == expected_available_memory
            + 2
            - headroom_correction
            - initial_device_charge
        )
    else:
        assert (
            available
            == 80 - headroom_correction - late_persistent_memory - initial_device_charge
        )
    # The activation peak stays activation-only (the repeatable allocator
    # headroom, seven bytes here, or the single profile's five); post-capture
    # recommendations add measured graph memory to it, and the admission
    # budget subtracts the estimate separately.
    assert worker.peak_activation_memory == 5 + headroom_correction
    assert worker.total_consumed == 10 + late_persistent_memory
    assert worker.cudagraph_memory_estimate == (
        cudagraph_estimate if profile_graphs else 0
    )


@pytest.mark.parametrize("estimated_gib", [0, 4])
@pytest.mark.parametrize("measured_gib", [3, 7])
@pytest.mark.parametrize("initial_device_charge", [0, 512 * (1 << 20)])
def test_post_capture_recommendation_counts_measured_graph_memory_once(
    monkeypatch, estimated_gib, measured_gib, initial_device_charge
):
    """The saved KV budget uses measured graph storage, not its estimate."""
    compilation = SimpleNamespace(
        mode=gpu_worker.CompilationMode.NONE,
        compilation_time=0.0,
        encoder_compilation_time=0.0,
    )
    worker = SimpleNamespace(
        vllm_config=SimpleNamespace(compilation_config=compilation),
        compilation_config=compilation,
        model_runner=SimpleNamespace(
            lora_config=None,
            maybe_remove_all_loras=lambda config: None,
            capture_model=lambda: measured_gib * GiB_bytes,
        ),
        model_config=SimpleNamespace(enforce_eager=False, seed=0),
        cache_config=SimpleNamespace(
            kv_cache_memory_bytes=None, gpu_memory_utilization=0.9
        ),
        init_snapshot=SimpleNamespace(
            free_memory=100 * GiB_bytes, total_memory=100 * GiB_bytes
        ),
        requested_memory=90 * GiB_bytes,
        initial_device_memory_charge=initial_device_charge,
        total_consumed=10 * GiB_bytes,
        peak_activation_memory=5 * GiB_bytes,
        cudagraph_memory_estimate=estimated_gib * GiB_bytes,
        available_kv_cache_memory_bytes=(75 - estimated_gib) * GiB_bytes,
        _b12x_session=None,
        use_v2_model_runner=False,
        observability_config=SimpleNamespace(
            jit_monitor_mode="off", jit_monitor_verbose=False
        ),
    )
    saved = []
    monkeypatch.setattr(
        gpu_worker, "maybe_save_startup_plan", lambda w, budget: saved.append(budget)
    )
    monkeypatch.setattr(
        gpu_worker, "get_pp_group", lambda: SimpleNamespace(is_last_rank=False)
    )
    for name in (
        "kernel_warmup",
        "set_random_seed",
        "freeze_gc_heap",
        "maybe_attach_gc_debug_callback",
        "enable_gpu_sync_check",
        "set_torch_threads_for_runtime",
    ):
        monkeypatch.setattr(gpu_worker, name, lambda *args: None)
    monkeypatch.setattr("vllm.utils.jit_monitor.activate", lambda **kwargs: None)

    gpu_worker.Worker._compile_or_warm_up_model_after_preparation(worker)

    assert saved == [
        (90 - 10 - 5 - measured_gib) * GiB_bytes
        - 150 * (1 << 20)
        - initial_device_charge
    ]


@pytest.mark.parametrize("dcp_size", [1, 4])
@pytest.mark.parametrize("external_shortfall", [False, True])
def test_dcp_startup_charges_communicators_inside_requested_memory(
    monkeypatch,
    dcp_size,
    external_shortfall,
):
    import torch

    from vllm.utils.mem_utils import MemorySnapshot
    from vllm.v1.worker import gpu_model_runner

    total = 100 * GiB_bytes
    requested = 97.5 * GiB_bytes
    before_free = int(requested + (150 if not external_shortfall else -10) * 1024**2)
    owned = 350 * 1024**2
    before = MemorySnapshot(
        device="cuda:0", auto_measure=False, total_memory=total, free_memory=before_free
    )
    after = MemorySnapshot(
        device="cuda:0",
        auto_measure=False,
        total_memory=total,
        free_memory=before_free - owned,
    )
    snapshots = iter([before, after] if dcp_size > 1 else [after])
    parallel = SimpleNamespace(
        distributed_executor_backend="external_launcher",
        data_parallel_backend="mp",
        assigned_physical_gpu_ids=None,
        enable_dbo=False,
        decode_context_parallel_size=dcp_size,
    )
    config = SimpleNamespace(parallel_config=parallel)
    worker = SimpleNamespace(
        device_config=SimpleNamespace(device_type="cuda"),
        parallel_config=parallel,
        vllm_config=config,
        local_rank=0,
        rank=1,
        distributed_init_method="unused",
        use_v2_model_runner=False,
        model_config=SimpleNamespace(dtype=torch.bfloat16, seed=0),
        cache_config=SimpleNamespace(gpu_memory_utilization=0.975),
    )
    monkeypatch.setattr(gpu_worker, "MemorySnapshot", lambda **kw: next(snapshots))
    monkeypatch.setattr(
        gpu_worker, "init_worker_distributed_environment", lambda *a: None
    )
    monkeypatch.setattr(gpu_worker, "set_random_seed", lambda *a: None)
    monkeypatch.setattr(gpu_worker, "init_workspace_manager", lambda *a: None)
    monkeypatch.setattr(gpu_worker, "_num_workspace_lanes", lambda *a: 1)
    monkeypatch.setattr(
        gpu_worker,
        "current_platform",
        SimpleNamespace(
            logical_device_id_to_visible_device_id=lambda rank: rank,
            check_if_supports_dtype=lambda dtype: None,
            dist_backend="nccl",
        ),
    )
    monkeypatch.setattr(torch.accelerator, "device_count", lambda: 8)
    monkeypatch.setattr(torch.accelerator, "set_device_index", lambda *a: None)
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: None)
    monkeypatch.setattr(gpu_model_runner, "GPUModelRunner", lambda *a: None)
    if external_shortfall or dcp_size == 1:
        with pytest.raises(ValueError, match="Free memory on device"):
            gpu_worker.Worker.init_device(worker)
    else:
        gpu_worker.Worker.init_device(worker)
        assert worker.requested_memory == requested
        assert worker.distributed_init_memory == owned
        # The profiler's baseline includes communicator bytes once in the
        # free-memory delta used to subtract non-KV storage from the request.
        assert worker.init_snapshot is before
        free_after_model = after.free_memory - 60 * GiB_bytes
        consumed = worker.init_snapshot.free_memory - free_after_model
        assert consumed == 60 * GiB_bytes + owned
        assert worker.requested_memory - consumed == requested - 60 * GiB_bytes - owned
    assert worker.initial_device_memory_charge == (
        total - before_free if dcp_size > 1 else 0
    )


def test_dcp1_does_not_inspect_projected_prefill_resources():
    worker = SimpleNamespace(
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        model_runner=None,
    )
    assert not gpu_worker._has_dcp_prefill_warmup(worker)


@pytest.mark.parametrize("mechanism", ["projected", "query_bmm", "ckv", "disabled"])
def test_eager_prefill_warmup_gate_uses_configured_mechanisms(mechanism):
    layer = SimpleNamespace(
        dcp_manager=SimpleNamespace(
            prefill_warmup_key=("projected",) if mechanism == "projected" else None
        ),
        _prefill_query_bmm_module=object() if mechanism == "query_bmm" else None,
        impl=SimpleNamespace(_ckv_gather_enabled=mechanism == "ckv"),
    )
    worker = SimpleNamespace(
        parallel_config=SimpleNamespace(decode_context_parallel_size=4),
        model_runner=SimpleNamespace(
            get_model=lambda: SimpleNamespace(modules=lambda: [layer])
        ),
    )
    assert gpu_worker._has_dcp_prefill_warmup(worker) == (mechanism != "disabled")


@pytest.mark.parametrize("resolves_kernels", [False, True])
def test_explicit_kv_budget_initializes_opted_prefill_before_return(
    monkeypatch, resolves_kernels
):
    events = []
    layer = SimpleNamespace(_prefill_query_bmm_module=object())

    def profile(prepare):
        prepare()
        events.append("profile")

    worker = SimpleNamespace(
        parallel_config=SimpleNamespace(decode_context_parallel_size=4),
        cache_config=SimpleNamespace(kv_cache_memory_bytes=4096),
        init_snapshot=SimpleNamespace(free_memory=8192),
        model_config=SimpleNamespace(multimodal_config=None),
        model_runner=SimpleNamespace(
            get_model=lambda: SimpleNamespace(modules=lambda: [layer]),
            profile_run=profile,
        ),
        _prepare_b12x_profile_state=lambda: events.append("prepare"),
        _release_b12x_profile_state=lambda: events.append("release"),
        compilation_config=SimpleNamespace(cudagraph_capture_sizes=[]),
        vllm_config=SimpleNamespace(
            compilation_config=SimpleNamespace(cudagraph_capture_sizes=[])
        ),
    )
    monkeypatch.setattr(gpu_worker, "maybe_apply_startup_plan", lambda _: None)

    def warmup(worker, sizes):
        assert sizes == []
        events.append("warmup")
        return resolves_kernels

    def budget(requested, *args):
        events.append("admit")
        return requested

    monkeypatch.setattr(gpu_worker, "warmup_dcp_prefill", warmup)
    monkeypatch.setattr(gpu_worker, "reserve_mm_ipc_gpu_memory", budget)
    assert gpu_worker.Worker.determine_available_memory(worker) == 4096
    assert events == [
        "prepare",
        "profile",
        "release",
        "warmup",
        *(["prepare", "profile", "release"] if resolves_kernels else []),
        "admit",
    ]
