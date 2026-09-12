"""Single-GPU check of the stable copy-in CUDA-graph mechanism.

This does not reproduce the multi-rank collective bug. It verifies on the
assigned gfx950 that separately captured graphs can copy graph-specific inputs
into one stable address and produce the expected numerical results.
"""

import json

import torch

device = torch.device("cuda:0")
stable = torch.empty(1024, dtype=torch.bfloat16, device=device)
sources = [torch.full_like(stable, value) for value in (1.0, 2.0, 4.0, 8.0)]
outputs = [torch.empty((), dtype=torch.float32, device=device) for _ in sources]
graphs = []

for source, output in zip(sources, outputs):
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        stable.copy_(source)
        torch.sum(stable, dim=(0,), dtype=torch.float32, out=output)
    graphs.append(graph)

observed = []
for graph, output in zip(reversed(graphs), reversed(outputs)):
    graph.replay()
    observed.append(output.item())

expected = [8192.0, 4096.0, 2048.0, 1024.0]
torch.cuda.synchronize()
assert observed == expected, (observed, expected)
print(
    json.dumps(
        {
            "device": torch.cuda.get_device_name(0),
            "architecture": torch.cuda.get_device_properties(0).gcnArchName,
            "stable_address": stable.data_ptr(),
            "replay_order": [3, 2, 1, 0],
            "observed_sums": observed,
            "expected_sums": expected,
        },
        indent=2,
    )
)
