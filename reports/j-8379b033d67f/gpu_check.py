#!/usr/bin/env python3
"""Small numerical check for the assigned GPU; not a model-serving fixture."""

import torch


assert torch.cuda.device_count() == 1
props = torch.cuda.get_device_properties(0)
assert props.gcnArchName.startswith("gfx950")

a_cpu = torch.arange(1, 17, dtype=torch.float32).reshape(4, 4)
b_cpu = torch.arange(17, 33, dtype=torch.float32).reshape(4, 4)
expected = a_cpu @ b_cpu
actual = a_cpu.cuda() @ b_cpu.cuda()
torch.cuda.synchronize()
torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)

print("torch", torch.__version__, "hip", torch.version.hip)
print("device_count", torch.cuda.device_count())
print("device_name", props.name)
print("gcnArchName", props.gcnArchName)
print("matrix_result", actual.cpu().tolist())
print("independent_cpu_reference_equal", True)
