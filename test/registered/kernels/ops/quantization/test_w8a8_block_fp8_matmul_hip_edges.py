import pytest
import torch
from unittest import mock

from sglang.kernels.ops.quantization import fp8_kernel
from sglang.srt.utils.common import is_gfx942_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci


register_amd_ci(est_time=20, stage="jit-kernel-unit", runner_config="amd")


_RUNNABLE = is_hip() and is_gfx942_supported()
_BLOCK_SIZE = [128, 128]
_OUTPUT_DTYPE = torch.bfloat16
_SENTINEL = -12352.0


def _make_case(M, N, K, mode):
    torch.manual_seed(1234 + M + N + K)
    device = torch.device("cuda")
    quant_dtype = fp8_kernel.fp8_dtype

    A_quant = (
        torch.randn(M, K, device=device, dtype=torch.float32)
        .clamp(-1, 1)
        .to(quant_dtype)
    )
    B_quant = (
        torch.randn(N, K, device=device, dtype=torch.float32)
        .clamp(-1, 1)
        .to(quant_dtype)
    )

    A_scale = torch.rand(M, (K + 127) // 128, device=device, dtype=torch.float32) * 0.5
    B_scale = (
        torch.rand(
            (N + 127) // 128,
            (K + 127) // 128,
            device=device,
            dtype=torch.float32,
        )
        * 0.5
    )

    if mode == "zero":
        A_quant[:, :128] = 0
        B_quant[:128, :128] = 0
    elif mode == "outlier":
        A_quant[:, 128:256] = torch.finfo(quant_dtype).max
        B_quant[128:256, 128:256] = torch.finfo(quant_dtype).max
        A_scale[:, 1] = 1.0 / torch.finfo(quant_dtype).max
        B_scale[1, 1] = 1.0 / torch.finfo(quant_dtype).max

    A_dequant = A_quant.float() * A_scale.repeat_interleave(128, dim=1)[:M, :K]
    B_dequant = B_quant.float() * B_scale.repeat_interleave(
        128, dim=0
    ).repeat_interleave(128, dim=1)[:N, :K]
    reference = A_dequant.to(_OUTPUT_DTYPE) @ B_dequant.to(_OUTPUT_DTYPE).T

    return A_quant, B_quant, A_scale, B_scale, reference


@pytest.mark.skipif(not _RUNNABLE, reason="requires HIP gfx942 (MI300X/MI325X)")
@pytest.mark.parametrize(
    "M,N,K,mode",
    [
        (1, 1, 128, "normal"),
        (7, 129, 129, "normal"),
        (63, 128, 255, "normal"),
        (256, 1088, 512, "normal"),
        (256, 1088, 1024, "normal"),
        (7, 129, 256, "zero"),
        (7, 129, 256, "outlier"),
    ],
)
def test_w8a8_block_fp8_matmul_hip_edges(M, N, K, mode):
    assert fp8_kernel.fp8_dtype == torch.float8_e4m3fnuz
    assert fp8_kernel.fp8_max == 224.0

    A_quant, B_quant, A_scale, B_scale, reference = _make_case(M, N, K, mode)
    assert A_scale.shape == (M, (K + 127) // 128)
    assert B_scale.shape == ((N + 127) // 128, (K + 127) // 128)

    default_config = {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128}
    selected_kernel = fp8_kernel.select_w8a8_block_fp8_matmul_kernel(
        M, N, default_config
    )
    assert selected_kernel is fp8_kernel._w8a8_block_fp8_matmul_unrolledx4

    original_prepare = fp8_kernel.prepare_block_fp8_matmul_inputs

    def prepare_with_sentinel(*args, **kwargs):
        prepared_M, prepared_N, prepared_K, output = original_prepare(*args, **kwargs)
        output.fill_(_SENTINEL)
        return prepared_M, prepared_N, prepared_K, output

    with mock.patch.object(
        fp8_kernel, "get_w8a8_block_fp8_configs", return_value=None
    ), mock.patch.object(
        fp8_kernel, "prepare_block_fp8_matmul_inputs", prepare_with_sentinel
    ):
        output = fp8_kernel.w8a8_block_fp8_matmul_triton(
            A_quant,
            B_quant,
            A_scale,
            B_scale,
            _BLOCK_SIZE,
            output_dtype=_OUTPUT_DTYPE,
        )

    assert not torch.any(output == _SENTINEL)
    torch.testing.assert_close(output, reference, atol=0.5, rtol=1e-4)
