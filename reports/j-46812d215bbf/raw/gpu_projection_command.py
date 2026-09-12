from types import SimpleNamespace

import torch

from sglang.srt.models.dspark import project_through_lm_head

assert torch.cuda.is_available()
device = torch.device("cuda:0")
torch.manual_seed(35354)
hidden = torch.randn(35 * 7, 512, device=device, dtype=torch.float32)
logical_weight = torch.randn(1024, 512, device=device, dtype=torch.float32)


class ModelOptFp4LinearMethod:
    quant_mode = "w4a16"

    def apply(self, layer, x, bias):
        assert bias is None
        return x @ logical_weight.T


head = SimpleNamespace(
    weight=torch.empty(1024, 256, device=device, dtype=torch.uint8),
    quant_method=ModelOptFp4LinearMethod(),
    weight_scale_interleaved=torch.empty(0, device=device),
    alpha=torch.empty(0, device=device),
    input_size_per_partition=512,
    output_size_per_partition=1024,
)

try:
    hidden @ head.weight.T
except RuntimeError as exc:
    print("legacy_failure=", str(exc).splitlines()[0])
else:
    raise AssertionError("legacy packed-storage matmul unexpectedly succeeded")

actual = project_through_lm_head(hidden, head)
reference = hidden @ logical_weight.T
max_abs_error = (actual - reference).abs().max().item()
print("device=", torch.cuda.get_device_name(0))
print("gfx=", torch.cuda.get_device_properties(0).gcnArchName)
print(
    "hidden_shape=",
    tuple(hidden.shape),
    "packed_shape=",
    tuple(head.weight.shape),
    "output_shape=",
    tuple(actual.shape),
)
print("max_abs_error=", max_abs_error)
torch.testing.assert_close(actual, reference, rtol=0, atol=0)
