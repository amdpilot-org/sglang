import inspect
from unittest.mock import patch

import torch

from sglang.srt.layers.quantization import fp8_utils
from sglang.srt.platforms.cuda import CudaSRTPlatform
from sglang.srt.platforms.device_mixin import DeviceCapability


def capability_probe():
    platform = CudaSRTPlatform()
    values = {}
    for capability in ((7, 5), (8, 0), (8, 9), (9, 0)):
        with patch.object(
            platform,
            "get_device_capability",
            return_value=DeviceCapability(*capability),
        ):
            values[str(capability)] = platform.supports_fp8()
    print("capability_probe", values)


def forced_unsupported_per_tensor():
    torch.manual_seed(8128)
    m, k, n = 5, 16, 7
    input_tensor = torch.randn(m, k, dtype=torch.float16)
    input_scale = torch.tensor([0.25], dtype=torch.float32)
    weight_scale = torch.tensor(0.5, dtype=torch.float32)
    qinput = (input_tensor / input_scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    weight = torch.randn(k, n).to(torch.float8_e4m3fn)
    expected = (
        torch.mm(qinput.float(), weight.float()) * input_scale * weight_scale
    ).to(torch.float16)
    with (
        patch.object(fp8_utils, "_is_hip", False),
        patch.object(fp8_utils, "static_quant_fp8", return_value=(qinput, input_scale)),
        patch.object(
            torch,
            "_scaled_mm",
            side_effect=RuntimeError("forced unsupported native FP8 GEMM"),
        ),
    ):
        try:
            actual = fp8_utils.apply_fp8_linear(
                input_tensor,
                weight,
                weight_scale,
                input_scale=input_scale,
                cutlass_fp8_supported=False,
                pad_output=False,
            )
        except Exception as exc:
            print("forced_unsupported_per_tensor", type(exc).__name__, str(exc))
            return False
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
    print("forced_unsupported_per_tensor PASS max_abs", (actual - expected).abs().max().item())
    return True


def scalar_and_channelwise_direct_fallback():
    torch.manual_seed(937)
    for m in (1, 4):
        k, n = 8, 3
        qinput = torch.randn(m, k).to(torch.float8_e4m3fn)
        weight = torch.randn(k, n).to(torch.float8_e4m3fn)
        for scale_kind in ("scalar", "channelwise"):
            x_scale = torch.tensor(0.125) if scale_kind == "scalar" else torch.rand(m, 1) + 0.1
            weight_scale = torch.tensor(0.25) if scale_kind == "scalar" else torch.rand(n, 1) + 0.1
            bias = torch.randn(n)
            expected = (
                torch.mm(qinput.float(), weight.float())
                * x_scale
                * weight_scale.t()
                + bias
            ).to(torch.float16)
            with patch.object(
                torch,
                "_scaled_mm",
                side_effect=RuntimeError("must not call native scaled_mm"),
            ):
                try:
                    actual = fp8_utils._apply_fallback_scaled_mm(
                        qinput,
                        weight,
                        x_scale,
                        weight_scale,
                        (m, k),
                        (m, n),
                        bias,
                        torch.float16,
                    )
                except Exception as exc:
                    print("direct_fallback", m, scale_kind, type(exc).__name__, str(exc))
                    continue
            torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
            print("direct_fallback", m, scale_kind, "PASS")


print("torch", torch.__version__, "hip", torch.version.hip)
print("fp8_utils_source", inspect.getsourcefile(fp8_utils))
print("cuda_platform_source", inspect.getsourcefile(CudaSRTPlatform))
capability_probe()
forced_unsupported_per_tensor()
scalar_and_channelwise_direct_fallback()
