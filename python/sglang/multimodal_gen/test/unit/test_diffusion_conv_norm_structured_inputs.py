from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.models.upsampler.latent_upsampler import ResBlock


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Structured diffusion conv/norm coverage requires CUDA",
)


def _structured_input(case: str, shape: tuple[int, int, int, int]) -> torch.Tensor:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(55161)
    if case == "zeros":
        return torch.zeros(shape, device="cuda", dtype=torch.bfloat16)
    if case == "tiny":
        values = torch.rand(shape, device="cuda", generator=generator)
        return values.mul_(2e-8).sub_(1e-8).to(torch.bfloat16)
    if case == "mixed_magnitudes":
        count = torch.tensor(shape, device="cuda").prod().item()
        values = torch.logspace(
            -4,
            4,
            count,
            device="cuda",
            dtype=torch.float32,
        ).reshape(shape)
        signs = torch.where(
            torch.rand(shape, device="cuda", generator=generator) < 0.5,
            -1.0,
            1.0,
        )
        return values.mul_(signs).to(torch.bfloat16)
    if case == "cancellation":
        half = torch.rand(
            (shape[0], shape[1], shape[2], shape[3] // 2),
            device="cuda",
            generator=generator,
        ).sub_(0.5)
        return torch.cat((half, -half), dim=3).to(torch.bfloat16)
    if case == "skewed_sparse":
        values = torch.rand(shape, device="cuda", generator=generator)
        mask = values.lt(0.001)
        return values.mul_(mask.float().mul_(1000.0)).to(torch.bfloat16)
    raise AssertionError(f"unknown case {case}")


def _manual_group_norm_silu(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    num_groups: int,
    eps: float,
    apply_silu: bool,
) -> torch.Tensor:
    original_dtype = x.dtype
    x_float = x.float()
    batch, channels = x_float.shape[:2]
    grouped = x_float.reshape(batch, num_groups, -1)
    mean = grouped.mean(dim=-1, keepdim=True)
    variance = grouped.var(dim=-1, unbiased=False, keepdim=True)
    normalized = (grouped - mean) * torch.rsqrt(variance + eps)
    normalized = normalized.reshape_as(x_float)
    normalized = normalized * weight.float().reshape(1, -1, *([1] * (x.dim() - 2)))
    normalized = normalized + bias.float().reshape(1, -1, *([1] * (x.dim() - 2)))
    if apply_silu:
        normalized = F.silu(normalized)
    return normalized.to(original_dtype)


def _independent_resblock_reference(block: ResBlock, x: torch.Tensor) -> torch.Tensor:
    residual = x
    conv1 = F.conv2d(
        x,
        block.conv1.weight,
        bias=block.conv1.bias,
        stride=block.conv1.stride,
        padding=block.conv1.padding,
    )
    normalized1 = _manual_group_norm_silu(
        conv1,
        block.norm1.weight,
        block.norm1.bias,
        block.norm1.num_groups,
        block.norm1.eps,
        apply_silu=True,
    )
    conv2 = F.conv2d(
        normalized1,
        block.conv2.weight,
        bias=block.conv2.bias,
        stride=block.conv2.stride,
        padding=block.conv2.padding,
    )
    normalized2 = _manual_group_norm_silu(
        conv2,
        block.norm2.weight,
        block.norm2.bias,
        block.norm2.num_groups,
        block.norm2.eps,
        apply_silu=False,
    )
    return F.silu(normalized2 + residual).to(x.dtype)


@requires_cuda
@pytest.mark.parametrize(
    "case",
    [
        "zeros",
        "tiny",
        "mixed_magnitudes",
        "cancellation",
        "skewed_sparse",
    ],
)
def test_resblock_structured_inputs_cuda(case: str):
    torch.manual_seed(55161)
    shape = (2, 64, 32, 32)
    block = ResBlock(channels=64, dims=2).to(
        device=torch.device("cuda"), dtype=torch.bfloat16
    )
    block.eval()
    x = _structured_input(case, shape)

    with torch.no_grad():
        actual = block(x)
        reference = _independent_resblock_reference(block, x)

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, reference, atol=7e-2, rtol=2e-2)
