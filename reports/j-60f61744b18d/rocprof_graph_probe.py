#!/usr/bin/env python3
"""Check rocprofiler-sdk dispatch tracing for HIP graph replay on one GPU."""

import json
import math
import time

import torch


torch.manual_seed(20260912)
a_cpu = torch.randn(64, 64, dtype=torch.float32)
b_cpu = torch.randn(64, 64, dtype=torch.float32)
reference = (a_cpu @ b_cpu).relu().sum().item()
a = a_cpu.cuda()
b = b_cpu.cuda()
out = torch.empty_like(a)
# Initialize the GEMM backend before stream capture.
torch.mm(a, b, out=out)
torch.cuda.synchronize()
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    torch.mm(a, b, out=out)
    out.relu_()

observed = []
for step in range(10):
    torch.cuda.nvtx.range_push(f"decode_step_{step}")
    graph.replay()
    torch.cuda.synchronize()
    observed.append(out.sum().item())
    torch.cuda.nvtx.range_pop()

result = {
    "torch": torch.__version__,
    "hip": torch.version.hip,
    "device": torch.cuda.get_device_name(),
    "reference_cpu_sum": reference,
    "observed_gpu_sums": observed,
    "max_abs_error": max(abs(x - reference) for x in observed),
    "all_finite": all(math.isfinite(x) for x in observed),
}
print(json.dumps(result, indent=2))
