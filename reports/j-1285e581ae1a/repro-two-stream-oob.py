import json

import torch


device = torch.device("cuda:0")
rows, cols = 128, 4096
source = torch.randn(rows, cols, device=device, dtype=torch.float32)
valid_index = torch.zeros((rows, 4), device=device, dtype=torch.int64)
valid_index[:, 1::2] = cols - 1
invalid_index = valid_index.clone()
invalid_index[0, 0] = -1
invalid_index[0, 1] = cols

stream0 = torch.cuda.Stream(device=device)
stream1 = torch.cuda.Stream(device=device)
torch.cuda.synchronize(device)
error = None
with torch.cuda.stream(stream0):
    invalid_output = torch.empty((rows, 4), device=device, dtype=torch.float32)
    try:
        invalid_output.copy_(torch.gather(source, 1, invalid_index))
    except RuntimeError as exc:
        error = str(exc)
with torch.cuda.stream(stream1):
    valid_output = torch.gather(source, 1, valid_index)

try:
    stream0.synchronize()
    stream1.synchronize()
except RuntimeError as exc:
    error = error or str(exc)

reference = torch.gather(source.detach().cpu(), 1, valid_index.detach().cpu())
valid_exact = False
if error is None:
    valid_exact = bool(torch.equal(valid_output.detach().cpu(), reference))
print(
    json.dumps(
        {
            "device": torch.cuda.get_device_name(device),
            "streams": 2,
            "independent_outputs": True,
            "invalid_indices": [-1, cols],
            "valid_boundary_indices": [0, cols - 1],
            "error": error,
            "valid_output_exact": valid_exact,
        },
        sort_keys=True,
    )
)
