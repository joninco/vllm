# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import torch

from vllm.model_executor.warmup import kernel_warmup


def test_resolve_flashinfer_autotune_file_default_layout(
    monkeypatch, tmp_path: Path
) -> None:
    fake_jit = SimpleNamespace(
        env=SimpleNamespace(
            FLASHINFER_WORKSPACE_DIR=Path("/flashinfer-cache/0.6.11.post2/103a")
        )
    )
    fake_flashinfer = SimpleNamespace(jit=fake_jit)
    monkeypatch.setitem(sys.modules, "flashinfer", fake_flashinfer)
    monkeypatch.setitem(sys.modules, "flashinfer.jit", fake_jit)
    monkeypatch.setattr(
        kernel_warmup, "aot_compile_hash_factors", lambda _: ["env-hash", "config-hash"]
    )
    monkeypatch.setattr(kernel_warmup.envs, "VLLM_CACHE_ROOT", str(tmp_path))
    monkeypatch.setattr(kernel_warmup.envs, "VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR", None)

    runner = SimpleNamespace(vllm_config=SimpleNamespace())
    cache_hash = sha256(str(["env-hash", "config-hash"]).encode()).hexdigest()

    path = kernel_warmup._resolve_flashinfer_autotune_file(runner)

    assert path == (
        tmp_path
        / "flashinfer_autotune_cache"
        / "0.6.11.post2"
        / "103a"
        / cache_hash
        / "autotune_configs.json"
    )
    assert path.parent.is_dir()


def test_resolve_flashinfer_autotune_file_uses_override_dir(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        kernel_warmup.envs, "VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR", str(tmp_path)
    )
    monkeypatch.setattr(
        kernel_warmup, "aot_compile_hash_factors", lambda _: ["env-hash", "config-hash"]
    )

    runner = SimpleNamespace(vllm_config=SimpleNamespace())
    cache_hash = sha256(str(["env-hash", "config-hash"]).encode()).hexdigest()

    path = kernel_warmup._resolve_flashinfer_autotune_file(runner)

    assert path == tmp_path / cache_hash / "autotune_configs.json"


def _flashinfer_autotune_worker(model, *, attn_groups=None):
    runner = SimpleNamespace(
        attn_groups=attn_groups or [],
        is_pooling_model=True,
    )
    return SimpleNamespace(
        get_model=lambda: model,
        model_runner=runner,
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8),
        vllm_config=SimpleNamespace(
            compilation_config=SimpleNamespace(cudagraph_capture_sizes=[]),
            kernel_config=SimpleNamespace(enable_flashinfer_autotune=True),
        ),
    )


def _patch_flashinfer_autotune_deps(monkeypatch):
    calls = []
    monkeypatch.setattr(kernel_warmup, "deepseek_v4_mhc_warmup", lambda *a, **k: None)
    monkeypatch.setattr(kernel_warmup, "has_flashinfer", lambda: True)
    monkeypatch.setattr(
        kernel_warmup.current_platform, "has_device_capability", lambda _: True
    )
    monkeypatch.setattr(
        kernel_warmup,
        "flashinfer_autotune",
        lambda runner: calls.append(runner),
    )
    return calls


def test_kernel_warmup_skips_flashinfer_autotune_without_flashinfer_kernels(
    monkeypatch,
) -> None:
    calls = _patch_flashinfer_autotune_deps(monkeypatch)
    worker = _flashinfer_autotune_worker(torch.nn.Linear(2, 2))

    kernel_warmup.kernel_warmup(worker)

    assert calls == []


def test_kernel_warmup_runs_flashinfer_autotune_for_attention(
    monkeypatch,
) -> None:
    calls = _patch_flashinfer_autotune_deps(monkeypatch)
    backend = SimpleNamespace(get_name=lambda: "FLASHINFER")
    group = SimpleNamespace(backend=backend)
    worker = _flashinfer_autotune_worker(
        torch.nn.Linear(2, 2),
        attn_groups=[[group]],
    )

    kernel_warmup.kernel_warmup(worker)

    assert calls == [worker.model_runner]


def test_kernel_warmup_runs_flashinfer_autotune_for_model_kernel(
    monkeypatch,
) -> None:
    calls = _patch_flashinfer_autotune_deps(monkeypatch)
    flashinfer_kernel_cls = type(
        "FlashInferKernel",
        (),
        {"__module__": "vllm.model_executor.kernels.linear.scaled_mm.flashinfer"},
    )

    class ModuleWithKernel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.quant_method = SimpleNamespace(
                scheme=SimpleNamespace(fp8_linear=flashinfer_kernel_cls())
            )

    worker = _flashinfer_autotune_worker(ModuleWithKernel())

    kernel_warmup.kernel_warmup(worker)

    assert calls == [worker.model_runner]
