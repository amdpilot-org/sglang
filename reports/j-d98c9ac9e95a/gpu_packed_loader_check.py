"""GPU numerical check for Qwen3.5's split AWQ checkpoint loader."""

from types import SimpleNamespace

import torch

from sglang.srt.layers.parameter import PackedvLLMParameter
from sglang.srt.models.qwen3_5 import Qwen3_5GatedDeltaNet


torch.manual_seed(31720)
logical = torch.randint(0, 16, (7, 48), device="cuda", dtype=torch.int32)
packed = torch.zeros((7, 6), device="cuda", dtype=torch.int32)
for lane in range(8):
    packed |= logical[:, lane::8] << (4 * lane)

param = PackedvLLMParameter(
    data=torch.empty_like(packed),
    input_dim=0,
    output_dim=1,
    packed_dim=1,
    packed_factor=8,
    weight_loader=lambda *args: None,
)
module = SimpleNamespace(output_sizes=[16, 8, 24, 32])
chunks = []


def original_loader(_param, chunk, shard_id):
    chunks.append((shard_id, chunk))


loader = Qwen3_5GatedDeltaNet._make_packed_weight_loader(module, original_loader)
loader(param, packed, (0, 1, 2))

unpacked_parts = []
for _, chunk in chunks:
    unpacked_parts.append(
        torch.stack([(chunk >> (4 * lane)) & 0xF for lane in range(8)], dim=2)
        .reshape(7, -1)
    )
loaded_logical = torch.cat(unpacked_parts, dim=1)
x = torch.randn((5, 7), device="cuda")
actual = x @ loaded_logical.float()
expected = x @ logical.float()
torch.testing.assert_close(actual, expected, rtol=0, atol=0)

print(
    "device",
    torch.cuda.get_device_name(0),
    torch.cuda.get_device_properties(0).gcnArchName,
)
print("chunk_widths", [chunk.shape[1] for _, chunk in chunks])
print("logical_equal", torch.equal(loaded_logical, logical))
print("matmul_max_abs_diff", (actual - expected).abs().max().item())
