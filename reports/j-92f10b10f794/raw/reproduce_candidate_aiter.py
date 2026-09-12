import os

os.environ["SGLANG_USE_AITER"] = "1"
os.environ["AITER_JIT_DIR"] = "/tmp/amdpilot-repo-j-92f10b10f794/cache/aiter"

import torch

from sglang.kernels.ops.gemm.bf16_fp32 import linear_bf16_fp32

torch.manual_seed(20260912)
device = "cuda"
print("device", torch.cuda.get_device_name(0))
for m in (1, 8, 16, 17, 64):
    x = torch.randn((m, 128), device=device, dtype=torch.bfloat16)
    weight = torch.randn((64, 128), device=device, dtype=torch.bfloat16)
    actual = linear_bf16_fp32(x, weight)
    reference = x.cpu().float() @ weight.cpu().float().t()
    actual_cpu = actual.cpu()
    rounded = reference.bfloat16().float()
    print(
        "M",
        m,
        "dtype",
        actual.dtype,
        "max_abs_vs_fp32",
        (actual_cpu - reference).abs().max().item(),
        "equals_fp32",
        torch.equal(actual_cpu, reference),
        "equals_bf16_rounded",
        torch.equal(actual_cpu, rounded),
        "all_values_bf16_representable",
        torch.equal(actual_cpu, actual_cpu.bfloat16().float()),
    )
