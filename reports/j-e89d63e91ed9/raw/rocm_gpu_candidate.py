import inspect

import torch

from sglang.srt.layers.quantization import fp8_utils


assert torch.cuda.is_available()
device = torch.device("cuda:0")
torch.manual_seed(20260912)
print("device", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
print("torch", torch.__version__, "hip", torch.version.hip)
print("source", inspect.getsourcefile(fp8_utils))

m, k, n = 32, 64, 32
input_tensor = torch.randn(m, k, device=device, dtype=torch.float16)
input_scale = torch.tensor([0.125], device=device, dtype=torch.float32)
weight_scale = torch.tensor(0.25, device=device, dtype=torch.float32)
qinput = (input_tensor / input_scale).clamp(-448, 448).to(torch.float8_e4m3fnuz)
weight = torch.randn(n, k, device=device).to(torch.float8_e4m3fnuz).t()

fallback = fp8_utils._apply_fallback_scaled_mm(
    qinput,
    weight,
    input_scale,
    weight_scale,
    (m, k),
    (m, n),
    None,
    torch.float16,
)
reference = (
    torch.mm(qinput.float(), weight.float()) * input_scale * weight_scale
).to(torch.float16)
torch.testing.assert_close(fallback, reference, rtol=1e-3, atol=1e-3)
print("software_fallback_gpu PASS", fallback.shape, fallback.dtype)

# ROCm intentionally retains its native per-tensor torch._scaled_mm route.
actual = fp8_utils.apply_fp8_linear(
    input_tensor,
    weight,
    weight_scale,
    input_scale=input_scale,
    cutlass_fp8_supported=False,
    pad_output=False,
)
torch.testing.assert_close(actual, reference, rtol=3e-2, atol=3e-2)
print("rocm_native_per_tensor PASS max_abs", (actual - reference).abs().max().item())
