"""Every-backend-vs-native parity for BaseFusedOp ops on real GPU (RFC #29630).

For each reworked fused op, run every backend eligible on this platform and
assert it matches the pure-torch ``forward_native`` reference within dtype
tolerance. New backends are picked up automatically.
"""

import pytest
import torch

from sglang.kernels.spec import KernelBackend
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=12, stage="extra-a", runner_config="1-gpu-small")

# torch_compile is native under the hood; skip it here (compile time dominates)
# -- it is exercised in the CPU lane.
_SKIP = {KernelBackend.TORCH, KernelBackend.TORCH_COMPILE}
_TOL = {
    torch.float16: dict(atol=1e-2, rtol=1e-2),
    torch.bfloat16: dict(atol=2e-2, rtol=2e-2),
}

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _eligible(op):
    return [
        b for b in op.available_backends() if b not in _SKIP and op.backend_eligible(b)
    ]


def _close(got, ref, dtype, msg):
    torch.testing.assert_close(got, ref, **_TOL[dtype], msg=msg)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(1, 4096), (128, 4096), (7, 2048)])
def test_rmsnorm(dtype, shape):
    from sglang.kernels.ops.layernorm import _RMSNORM

    torch.manual_seed(0)
    x = torch.randn(shape, dtype=dtype, device="cuda")
    w = torch.randn(shape[-1], dtype=dtype, device="cuda")
    ref = _RMSNORM.forward_native(x, w, 1e-6)
    for b in _eligible(_RMSNORM):
        _close(
            _RMSNORM.forward(x, w, 1e-6, backend=b), ref, dtype, f"rmsnorm {b.value}"
        )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(1, 4096), (128, 4096)])
def test_fused_add_rmsnorm(dtype, shape):
    from sglang.kernels.ops.layernorm import _FUSED_ADD_RMSNORM

    torch.manual_seed(0)
    x0 = torch.randn(shape, dtype=dtype, device="cuda")
    r0 = torch.randn(shape, dtype=dtype, device="cuda")
    w = torch.randn(shape[-1], dtype=dtype, device="cuda")
    x_ref, r_ref = x0.clone(), r0.clone()
    _FUSED_ADD_RMSNORM.forward_native(x_ref, r_ref, w, 1e-6)
    for b in _eligible(_FUSED_ADD_RMSNORM):
        x, r = x0.clone(), r0.clone()
        _FUSED_ADD_RMSNORM.forward(x, r, w, 1e-6, backend=b)
        _close(x, x_ref, dtype, f"fused_add {b.value} (normed)")
        _close(r, r_ref, dtype, f"fused_add {b.value} (residual)")


def _independent_fused_add_rmsnorm_reference(x, residual, weight, eps):
    accumulated = x.float() + residual.float()
    residual_out = accumulated.to(residual.dtype)
    variance = accumulated.pow(2).mean(dim=-1, keepdim=True)
    output = (
        accumulated * torch.rsqrt(variance + eps) * weight.float()
    ).to(x.dtype)
    return output, residual_out


def _assert_contract_close(actual, expected, dtype, message):
    tolerance = (
        dict(atol=2e-2, rtol=2e-2)
        if dtype in (torch.float16, torch.bfloat16)
        else dict(atol=1e-5, rtol=1e-5)
    )
    torch.testing.assert_close(actual, expected, **tolerance, msg=message)


@pytest.mark.parametrize(
    "case",
    [
        "contiguous_bf16",
        "row_strided_bf16",
        "aliased_bf16",
        "bf16_input_fp32_residual",
    ],
)
def test_fused_add_rmsnorm_public_ownership(case):
    from sglang.kernels.ops.layernorm import fused_add_rmsnorm

    torch.manual_seed(29630)
    shape = (7, 2048)
    input_dtype = torch.bfloat16
    residual_dtype = (
        torch.float32 if case == "bf16_input_fp32_residual" else torch.bfloat16
    )

    if case == "row_strided_bf16":
        input = torch.randn(shape[0], shape[1] * 2, dtype=input_dtype, device="cuda")[
            :, : shape[1]
        ]
        residual = torch.randn(
            shape[0], shape[1] * 2, dtype=residual_dtype, device="cuda"
        )[:, : shape[1]]
    elif case == "aliased_bf16":
        input = residual = torch.randn(shape, dtype=input_dtype, device="cuda")
    else:
        input = torch.randn(shape, dtype=input_dtype, device="cuda")
        residual = torch.randn(shape, dtype=residual_dtype, device="cuda")

    weight = torch.randn(shape[-1], dtype=input_dtype, device="cuda")
    input_before = input.clone()
    residual_before = residual.clone()
    input_pointer = input.data_ptr()
    residual_pointer = residual.data_ptr()

    result = fused_add_rmsnorm(input, residual, weight, 1e-6)
    torch.cuda.synchronize()

    output_reference, residual_reference = _independent_fused_add_rmsnorm_reference(
        input_before, residual_before, weight, 1e-6
    )
    if case == "aliased_bf16":
        residual_reference = output_reference

    assert result is None
    assert input.data_ptr() == input_pointer
    assert residual.data_ptr() == residual_pointer
    assert not torch.equal(input, input_before)
    assert not torch.equal(residual, residual_before)
    _assert_contract_close(
        input, output_reference, input.dtype, f"fused_add public output ({case})"
    )
    _assert_contract_close(
        residual,
        residual_reference,
        residual.dtype,
        f"fused_add public residual ({case})",
    )


@pytest.mark.parametrize(
    "case",
    [
        "mixed_residual_dtype",
        "nonunit_last_stride",
        "noncontiguous_weight",
        "three_dimensional",
    ],
)
def test_fused_add_rmsnorm_aiter_error_gates(case):
    from sglang.kernels.ops.layernorm import _FUSED_ADD_RMSNORM

    torch.manual_seed(29630)
    if case == "three_dimensional":
        input = torch.randn(2, 3, 64, dtype=torch.bfloat16, device="cuda")
    else:
        input = torch.randn(3, 64, dtype=torch.bfloat16, device="cuda")
        if case == "nonunit_last_stride":
            input = torch.randn(64, 3, dtype=torch.bfloat16, device="cuda").T

    residual_dtype = (
        torch.float32 if case == "mixed_residual_dtype" else torch.bfloat16
    )
    residual = torch.randn(input.shape, dtype=residual_dtype, device="cuda")
    weight = torch.randn(64, dtype=torch.bfloat16, device="cuda")
    if case == "noncontiguous_weight":
        weight = torch.randn(128, dtype=torch.bfloat16, device="cuda")[::2]

    input_before = input.clone()
    residual_before = residual.clone()

    with pytest.raises(RuntimeError, match="aiter fused_add_rmsnorm requires"):
        _FUSED_ADD_RMSNORM.forward(
            input, residual, weight, 1e-6, backend=KernelBackend.AITER
        )

    torch.testing.assert_close(input, input_before)
    torch.testing.assert_close(residual, residual_before)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gemma_rmsnorm(dtype):
    from sglang.kernels.ops.layernorm import _GEMMA_RMSNORM

    torch.manual_seed(0)
    x = torch.randn(64, 2048, dtype=dtype, device="cuda")
    w = torch.randn(2048, dtype=dtype, device="cuda")
    ref = _GEMMA_RMSNORM.forward_native(x, w, 1e-6)
    for b in _eligible(_GEMMA_RMSNORM):
        _close(
            _GEMMA_RMSNORM.forward(x, w, 1e-6, backend=b),
            ref,
            dtype,
            f"gemma {b.value}",
        )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gemma_fused_add_rmsnorm(dtype):
    from sglang.kernels.ops.layernorm import _GEMMA_FUSED_ADD_RMSNORM

    torch.manual_seed(0)
    x0 = torch.randn(64, 2048, dtype=dtype, device="cuda")
    r0 = torch.randn(64, 2048, dtype=dtype, device="cuda")
    w = torch.randn(2048, dtype=dtype, device="cuda")
    x_ref, r_ref = x0.clone(), r0.clone()
    _GEMMA_FUSED_ADD_RMSNORM.forward_native(x_ref, r_ref, w, 1e-6)
    for b in _eligible(_GEMMA_FUSED_ADD_RMSNORM):
        x, r = x0.clone(), r0.clone()
        _GEMMA_FUSED_ADD_RMSNORM.forward(x, r, w, 1e-6, backend=b)
        _close(x, x_ref, dtype, f"gemma_fused_add {b.value} (normed)")
        _close(r, r_ref, dtype, f"gemma_fused_add {b.value} (residual)")


@pytest.mark.parametrize(
    "op_attr", ["_SILU_AND_MUL", "_GELU_AND_MUL", "_GELU_TANH_AND_MUL"]
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(1, 8192), (128, 8192)])
def test_gated_activation(op_attr, dtype, shape):
    import sglang.kernels.ops.activation as act

    torch.manual_seed(0)
    op = getattr(act, op_attr)
    x = torch.randn(shape, dtype=dtype, device="cuda")
    ref = op.forward_native(x)
    for b in _eligible(op):
        _close(op.forward(x, backend=b), ref, dtype, f"{op.op} {b.value}")


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
