import json
import sys

import torch


trial = int(sys.argv[1])
torch.manual_seed(1285 + trial)
device = torch.device("cuda:0")
rows, cols = 2048, 4096
source = torch.randn(rows, cols, device=device, dtype=torch.float32)
pattern = torch.arange(4, device=device, dtype=torch.int64) % 2
index0 = torch.where(
    pattern == 0,
    torch.zeros_like(pattern),
    torch.full_like(pattern, cols - 1),
).repeat(rows, 1)
index1 = torch.where(
    pattern == 0,
    torch.full_like(pattern, cols - 1),
    torch.zeros_like(pattern),
).repeat(rows, 1)
reference0 = torch.gather(source.detach().cpu(), 1, index0.detach().cpu())
reference1 = torch.gather(source.detach().cpu(), 1, index1.detach().cpu())

stream0 = torch.cuda.Stream(device=device)
stream1 = torch.cuda.Stream(device=device)
start0 = torch.cuda.Event(enable_timing=True)
end0 = torch.cuda.Event(enable_timing=True)
start1 = torch.cuda.Event(enable_timing=True)
end1 = torch.cuda.Event(enable_timing=True)
torch.cuda.synchronize(device)

with torch.cuda.stream(stream0):
    start0.record()
    output0 = torch.gather(source, 1, index0)
    end0.record()

with torch.cuda.stream(stream1):
    start1.record()
    output1 = torch.gather(source, 1, index1)
    end1.record()

stream0.synchronize()
stream1.synchronize()
exact0 = bool(torch.equal(output0.detach().cpu(), reference0))
exact1 = bool(torch.equal(output1.detach().cpu(), reference1))
result = {
    "trial": trial,
    "device": torch.cuda.get_device_name(device),
    "streams": 2,
    "source_shape": [rows, cols],
    "index_shape": list(index0.shape),
    "boundary_indices": [0, cols - 1],
    "independent_outputs": True,
    "stream0_exact": exact0,
    "stream1_exact": exact1,
    "stream0_max_abs_diff": float((output0.detach().cpu() - reference0).abs().max()),
    "stream1_max_abs_diff": float((output1.detach().cpu() - reference1).abs().max()),
    "stream0_event_ms": start0.elapsed_time(end0),
    "stream1_event_ms": start1.elapsed_time(end1),
}
print(json.dumps(result, sort_keys=True))
if not exact0 or not exact1:
    raise SystemExit(1)
