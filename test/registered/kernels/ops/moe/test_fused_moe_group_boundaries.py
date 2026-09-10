import pytest
import torch
import triton.language as tl

from sglang.kernels.ops.moe.fused_moe_triton_kernels import invoke_fused_moe_kernel
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=10, stage="jit-kernel-unit", runner_config="amd")

if not torch.cuda.is_available():
    pytest.skip("Requires a GPU", allow_module_level=True)

DEVICE = "cuda"
FP8_DTYPE = torch.float8_e4m3fnuz if torch.version.hip else torch.float8_e4m3fn
SENTINEL = -9999.0
OUTPUT_BOUND = 2048.0


def _run_grouped_gemm(group_sizes):
    num_experts = len(group_sizes)
    output_size = 128
    input_size = 128
    total_rows = sum(group_sizes)
    block_size_m = 16

    activation_values = torch.tensor(
        [0, 1, -1, 2, -2], device=DEVICE, dtype=torch.float32
    )
    activation = activation_values[
        torch.arange(total_rows * input_size, device=DEVICE) % 5
    ].reshape(total_rows, input_size).to(FP8_DTYPE)
    activation_scale = torch.empty(
        (total_rows, 1), device=DEVICE, dtype=torch.float32
    )

    weight_values = torch.tensor(
        [1, -1, 2, -2, 0], device=DEVICE, dtype=torch.float32
    )
    weight = weight_values[
        torch.arange(num_experts * output_size * input_size, device=DEVICE) % 5
    ].reshape(num_experts, output_size, input_size).to(FP8_DTYPE)
    weight_scale = torch.empty(
        (num_experts, 1, 1), device=DEVICE, dtype=torch.float32
    )

    for row in range(total_rows):
        activation_scale[row, 0] = 0.25 * (row + 1)
    for expert in range(num_experts):
        weight_scale[expert, 0, 0] = 0.5 * (expert + 1)

    output = torch.full(
        (total_rows, output_size), SENTINEL, device=DEVICE, dtype=torch.float32
    )
    topk_weights = torch.ones(
        (total_rows, 1), device=DEVICE, dtype=torch.float32
    )
    topk_ids = torch.arange(
        total_rows, device=DEVICE, dtype=torch.int32
    ).reshape(total_rows, 1)

    sorted_token_ids = []
    expert_ids = []
    row_start = 0
    for expert, group_size in enumerate(group_sizes):
        row_end = row_start + group_size
        sorted_token_ids.extend(range(row_start, row_end))
        sorted_token_ids.extend([total_rows] * (block_size_m - group_size))
        expert_ids.append(expert)
        row_start = row_end

    sorted_token_ids = torch.tensor(
        sorted_token_ids, device=DEVICE, dtype=torch.int32
    )
    expert_ids = torch.tensor(expert_ids, device=DEVICE, dtype=torch.int32)
    num_tokens_post_padded = torch.tensor(
        [sorted_token_ids.numel()], device=DEVICE, dtype=torch.int32
    )
    config = {
        "BLOCK_SIZE_M": block_size_m,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 1,
    }

    invoke_fused_moe_kernel(
        activation,
        weight,
        None,
        output,
        activation_scale,
        weight_scale,
        None,
        topk_weights,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        False,
        1,
        config,
        tl.float32,
        True,
        False,
        False,
        False,
        False,
        [128, 128],
    )

    reference = torch.empty_like(output)
    row_start = 0
    for expert, group_size in enumerate(group_sizes):
        row_end = row_start + group_size
        if group_size:
            dequantized_activation = (
                activation[row_start:row_end].float()
                * activation_scale[row_start:row_end]
            )
            dequantized_weight = weight[expert].float() * weight_scale[expert]
            reference[row_start:row_end] = torch.mm(
                dequantized_activation, dequantized_weight.T
            )
        row_start = row_end

    return output, reference


@pytest.mark.parametrize(
    "group_sizes",
    [
        [0, 1, 2, 3],
        [0, 1, 0, 2, 1, 0, 3, 1],
    ],
)
@torch.inference_mode()
def test_prequantized_blockwise_fp8_group_boundaries(group_sizes):
    output, reference = _run_grouped_gemm(group_sizes)

    assert not torch.isinf(output).any()
    assert not (output == SENTINEL).any()
    assert output.abs().max() <= OUTPUT_BOUND
    torch.testing.assert_close(output, reference, rtol=0.0, atol=0.0)
