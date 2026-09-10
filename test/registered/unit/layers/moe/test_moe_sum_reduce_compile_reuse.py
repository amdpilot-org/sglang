import pytest
import torch

from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
    moe_sum_reduce_torch_compile,
)
from sglang.test.ci.ci_register import register_cuda_ci


register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")


SHAPES = (1, 2, 4, 8, 16, 32)
TOPK = 2
HIDDEN_SIZE = 64
ROUTED_SCALING_FACTOR = 0.375


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_compiled_combine_reuses_supported_shape_sequence():
    for num_tokens in SHAPES:
        torch.manual_seed(10_000 + num_tokens)
        x = (
            torch.randn(
                num_tokens, TOPK, HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16
            )
            * 0.1
        )
        out = torch.full(
            (num_tokens, HIDDEN_SIZE),
            float("nan"),
            device="cuda",
            dtype=torch.bfloat16,
        )
        input_sentinel = x.clone()
        output_address = out.data_ptr()

        moe_sum_reduce_torch_compile(x, out, ROUTED_SCALING_FACTOR)

        expected = (x.float().sum(dim=1) * ROUTED_SCALING_FACTOR).to(x.dtype)
        torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-2)
        assert torch.equal(x, input_sentinel)
        assert not torch.isnan(out).any()
        assert out.data_ptr() == output_address


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_compiled_combine_rejects_unsupported_contracts():
    x = torch.randn(2, TOPK, HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="dtype"):
        moe_sum_reduce_torch_compile(
            x,
            torch.empty(2, HIDDEN_SIZE, device="cuda", dtype=torch.float32),
            ROUTED_SCALING_FACTOR,
        )

    with pytest.raises(ValueError, match="shape"):
        moe_sum_reduce_torch_compile(
            x,
            torch.empty(2, HIDDEN_SIZE + 1, device="cuda", dtype=torch.bfloat16),
            ROUTED_SCALING_FACTOR,
        )

    with pytest.raises(ValueError, match="overlap"):
        aliased_output = x.reshape(-1)[: 2 * HIDDEN_SIZE].view(2, HIDDEN_SIZE)
        moe_sum_reduce_torch_compile(x, aliased_output, ROUTED_SCALING_FACTOR)

    with pytest.raises(ValueError, match="device"):
        moe_sum_reduce_torch_compile(
            x,
            torch.empty(2, HIDDEN_SIZE, device="cpu", dtype=torch.bfloat16),
            ROUTED_SCALING_FACTOR,
        )
