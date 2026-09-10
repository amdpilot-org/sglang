import pytest
import torch

from sglang.kernels.spec import KernelBackend
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=5, stage="jit-kernel-unit", runner_config="amd")


pytestmark = pytest.mark.skipif(
    torch.version.hip is None or not torch.cuda.is_available(),
    reason="AITer fused-add RMSNorm requires an available HIP GPU",
)


def test_fused_add_rmsnorm_aiter_residual_order():
    from sglang.kernels.ops.layernorm import _FUSED_ADD_RMSNORM

    torch.manual_seed(1184)
    shape = (16, 512)
    dtype = torch.bfloat16
    x0 = torch.randn(shape, dtype=dtype, device="cuda")
    residual0 = torch.randn(shape, dtype=dtype, device="cuda")
    weight = torch.randn(shape[-1], dtype=dtype, device="cuda")
    eps = 1e-6

    x_ref, residual_ref = x0.clone(), residual0.clone()
    _FUSED_ADD_RMSNORM.forward_native(x_ref, residual_ref, weight, eps)

    x, residual = x0.clone(), residual0.clone()
    _FUSED_ADD_RMSNORM.forward(
        x, residual, weight, eps, backend=KernelBackend.AITER
    )

    torch.testing.assert_close(
        x,
        x_ref,
        atol=2e-2,
        rtol=2e-2,
        msg="AITer normalized output differs from native reference",
    )
    torch.testing.assert_close(
        residual,
        residual_ref,
        atol=2e-2,
        rtol=2e-2,
        msg="AITer residual output differs from native reference",
    )
