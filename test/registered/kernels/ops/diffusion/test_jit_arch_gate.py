from types import SimpleNamespace

import pytest
import torch

from sglang.kernels.kda_kernels import residual_gate_add_jit
from sglang.kernels.ops.diffusion.rope import qknorm_rope_jit


@pytest.mark.parametrize("module", [qknorm_rope_jit, residual_gate_add_jit])
def test_diffusion_cuda_jit_rejects_pre_ampere_without_loading(module, monkeypatch):
    monkeypatch.setattr(module, "is_hip_runtime", lambda: False)
    monkeypatch.setattr(
        module, "get_jit_cuda_arch", lambda: SimpleNamespace(major=7, minor=5)
    )

    assert not module._is_diffusion_cuda_jit_supported()


@pytest.mark.parametrize("major,minor", [(8, 0), (9, 0), (12, 1)])
@pytest.mark.parametrize("module", [qknorm_rope_jit, residual_gate_add_jit])
def test_diffusion_cuda_jit_accepts_ampere_and_newer(
    module, monkeypatch, major, minor
):
    monkeypatch.setattr(module, "is_hip_runtime", lambda: False)
    monkeypatch.setattr(
        module,
        "get_jit_cuda_arch",
        lambda: SimpleNamespace(major=major, minor=minor),
    )

    assert module._is_diffusion_cuda_jit_supported()


@pytest.mark.parametrize("module", [qknorm_rope_jit, residual_gate_add_jit])
def test_diffusion_jit_arch_gate_does_not_treat_rocm_as_cuda_sm(module, monkeypatch):
    monkeypatch.setattr(module, "is_hip_runtime", lambda: True)
    monkeypatch.setattr(
        module, "get_jit_cuda_arch", lambda: SimpleNamespace(major=0, minor=0)
    )

    assert module._is_diffusion_cuda_jit_supported()


def test_qknorm_pre_ampere_gate_skips_jit_loader(monkeypatch):
    monkeypatch.setattr(qknorm_rope_jit, "_is_diffusion_cuda_jit_supported", lambda: False)
    monkeypatch.setattr(
        qknorm_rope_jit,
        "_jit_qknorm_rope_module",
        lambda *_args, **_kwargs: pytest.fail("pre-Ampere path attempted a JIT build"),
    )

    assert not qknorm_rope_jit._can_use_fused_qknorm_rope(
        128, 128, False, torch.float16, torch.float32, False, False, False
    )


def test_residual_gate_pre_ampere_rejects_before_inspecting_tensors(monkeypatch):
    monkeypatch.setattr(
        residual_gate_add_jit, "_is_diffusion_cuda_jit_supported", lambda: False
    )

    # Plain objects deliberately have none of the tensor attributes checked by
    # the normal eligibility path.  The architecture gate must run first.
    assert not residual_gate_add_jit.can_use_residual_gate_add_cuda(
        object(), object(), object()
    )
