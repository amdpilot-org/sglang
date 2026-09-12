"""Numerical shape/reference check for the reported Qwen3-VL merger adapter."""

import json

import torch


def check_projection(input_dim: int, output_dim: int, rank: int) -> dict:
    generator = torch.Generator().manual_seed(input_dim + output_dim)
    x_cpu = torch.randn(3, input_dim, generator=generator, dtype=torch.float32)
    a_cpu = torch.randn(rank, input_dim, generator=generator, dtype=torch.float32)
    b_cpu = torch.randn(output_dim, rank, generator=generator, dtype=torch.float32)
    reference = (x_cpu @ a_cpu.T) @ b_cpu.T

    x = x_cpu.to("cuda", dtype=torch.bfloat16)
    a = a_cpu.to("cuda", dtype=torch.bfloat16)
    b = b_cpu.to("cuda", dtype=torch.bfloat16)
    actual = ((x @ a.T) @ b.T).float().cpu()
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, reference, rtol=0.03, atol=10.0)

    return {
        "input_dim": input_dim,
        "output_dim": output_dim,
        "rank": rank,
        "max_abs_error": (actual - reference).abs().max().item(),
        "mean_abs_error": (actual - reference).abs().mean().item(),
        "max_reference_magnitude": reference.abs().max().item(),
    }


print(
    json.dumps(
        {
            "device": torch.cuda.get_device_name(0),
            "arch": torch.cuda.get_device_properties(0).gcnArchName,
            "linear_fc1": check_projection(4608, 4608, 32),
            "linear_fc2": check_projection(4608, 4096, 32),
        },
        indent=2,
    )
)
