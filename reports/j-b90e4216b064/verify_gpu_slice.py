import torch

from sglang.srt.managers.mm_utils import _slice_value


device = torch.device("cuda:0")
parent = torch.arange(32, dtype=torch.float32, device=device).reshape(8, 4)
split = _slice_value(parent, 2, 5)
expected = torch.arange(8, 20, dtype=torch.float32, device=device).reshape(3, 4)

torch.testing.assert_close(split, expected)
assert split.untyped_storage().data_ptr() == parent.untyped_storage().data_ptr()
print(f"device: {torch.cuda.get_device_name(0)}")
print(f"torch_hip: {torch.version.hip}")
print(f"values_match: {torch.equal(split, expected)}")
print("shares_parent_storage: true")
