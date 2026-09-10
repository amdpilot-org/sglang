import sys

import pytest
import torch

from sglang.kernels.ops.gemm import fp8_scaled_mm
from sglang.kernels.selector import get_kernel
from sglang.kernels.spec import KernelBackend
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(
    est_time=10,
    stage="base-b-kernel-unit",
    runner_config="1-gpu-large",
)
register_amd_ci(
    est_time=10,
    stage="jit-kernel-unit",
    runner_config="amd",
)


def _fp8_dtype():
    if torch.version.hip is not None:
        return torch.float8_e4m3fnuz
    return torch.float8_e4m3fn


def _make_case(num_rows, num_columns, num_contracted, bias_value, seed):
    torch.manual_seed(seed)
    input_dtype = _fp8_dtype()
    mat_a = ((torch.rand((num_rows, num_contracted), device="cuda") * 2 - 1) * 8).to(
        input_dtype
    )
    mat_b = (
        (torch.rand((num_columns, num_contracted), device="cuda") * 2 - 1) * 8
    ).to(input_dtype).t()
    scales_a = torch.full((num_rows,), 0.125, device="cuda", dtype=torch.float32)
    scales_b = torch.full((num_columns,), 0.125, device="cuda", dtype=torch.float32)
    if bias_value == "zero":
        bias = torch.zeros(num_columns, device="cuda", dtype=torch.bfloat16)
    else:
        bias = torch.linspace(-0.5, 0.5, num_columns, device="cuda").to(
            torch.bfloat16
        )
    return mat_a, mat_b, scales_a, scales_b, bias


def _reference(mat_a, mat_b, scales_a, scales_b, bias, out_dtype):
    product = torch.mm(mat_a.to(torch.float32), mat_b.to(torch.float32))
    product = product * scales_a.view(-1, 1) * scales_b.view(1, -1)
    return (product + bias.to(torch.float32)).to(out_dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA or HIP required")
@pytest.mark.parametrize("num_rows", [1, 3, 17])
@pytest.mark.parametrize("bias_value", ["zero", "finite"])
def test_fp8_scaled_mm_bias_epilogue(num_rows, bias_value):
    num_columns = 128
    num_contracted = 128
    mat_a, mat_b, scales_a, scales_b, bias = _make_case(
        num_rows, num_columns, num_contracted, bias_value, 29630 + num_rows
    )
    native_backend = get_kernel("gemm.fp8_scaled_mm", KernelBackend.TORCH)

    public_output = fp8_scaled_mm(
        mat_a,
        mat_b,
        scales_a,
        scales_b,
        torch.bfloat16,
        bias,
    )
    native_output = native_backend(
        mat_a,
        mat_b,
        scales_a,
        scales_b,
        torch.bfloat16,
        bias,
    )
    reference_output = _reference(
        mat_a, mat_b, scales_a, scales_b, bias, torch.bfloat16
    )

    assert public_output.shape == (num_rows, num_columns)
    assert native_output.shape == (num_rows, num_columns)
    assert public_output.dtype == torch.bfloat16
    assert native_output.dtype == torch.bfloat16
    torch.testing.assert_close(public_output, native_output, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(public_output, reference_output, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(native_output, reference_output, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA or HIP required")
@pytest.mark.parametrize(
    "invalid_kind",
    [
        "shape",
        "dtype",
    ],
)
def test_fp8_scaled_mm_rejects_invalid_bias(invalid_kind):
    num_rows = 3
    num_columns = 128
    num_contracted = 128
    mat_a, mat_b, scales_a, scales_b, _ = _make_case(
        num_rows, num_columns, num_contracted, "zero", 29630
    )
    native_backend = get_kernel("gemm.fp8_scaled_mm", KernelBackend.TORCH)
    if invalid_kind == "shape":
        invalid_bias = torch.full(
            (num_columns - 1,), 0.25, device="cuda", dtype=torch.bfloat16
        )
    else:
        invalid_bias = torch.full(
            (num_columns,), 0.25, device="cuda", dtype=torch.float16
        )

    with pytest.raises(RuntimeError):
        fp8_scaled_mm(
            mat_a,
            mat_b,
            scales_a,
            scales_b,
            torch.bfloat16,
            invalid_bias,
        )
    with pytest.raises(RuntimeError):
        native_backend(
            mat_a,
            mat_b,
            scales_a,
            scales_b,
            torch.bfloat16,
            invalid_bias,
        )


@pytest.mark.skipif(
    torch.version.hip is None or not torch.cuda.is_available(),
    reason="HIP FP16 output restriction required",
)
def test_hip_fp8_scaled_mm_rejects_float16_output():
    num_rows = 3
    num_columns = 128
    num_contracted = 128
    mat_a, mat_b, scales_a, scales_b, bias = _make_case(
        num_rows, num_columns, num_contracted, "finite", 29630
    )
    native_backend = get_kernel("gemm.fp8_scaled_mm", KernelBackend.TORCH)

    with pytest.raises(RuntimeError, match="BFloat16"):
        fp8_scaled_mm(
            mat_a,
            mat_b,
            scales_a,
            scales_b,
            torch.float16,
            bias.to(torch.float16),
        )
    with pytest.raises(RuntimeError, match="BFloat16"):
        native_backend(
            mat_a,
            mat_b,
            scales_a,
            scales_b,
            torch.float16,
            bias.to(torch.float16),
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
