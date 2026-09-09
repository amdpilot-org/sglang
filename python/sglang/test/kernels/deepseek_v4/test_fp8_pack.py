import pathlib

import pytest
import torch

import sglang
from sglang.kernels.jit.utils import load_jit
from sglang.kernels.jit.utils.common import is_hip_runtime


def _probe_module():
    probe = (
        pathlib.Path(sglang.__file__).resolve().parent
        / "kernels/jit/csrc/deepseek_v4/fp8_pack_probe.cuh"
    )
    return load_jit(
        "dsv4_fp8_pack_probe_regression",
        cuda_files=[str(probe)],
        cuda_wrappers=[("run", "fp8_pack_probe::run")],
    )


def _expected_bytes(values, policy_max):
    values = values.clamp(min=-policy_max, max=policy_max)
    return values.to(torch.float8_e4m3fnuz).view(torch.uint8)


@pytest.mark.parametrize(
    "values",
    [
        [1.0, -1.0, 128.0, 160.0, 224.0, -1e-8],
        [0.0, -0.0, 1e-8, -1e-8],
        [
            2.0**-11,
            torch.nextafter(torch.tensor(2.0**-11), torch.tensor(0.0)).item(),
            -2.0**-11,
            torch.nextafter(torch.tensor(-2.0**-11), torch.tensor(0.0)).item(),
            torch.nextafter(torch.tensor(2.0**-11), torch.tensor(float("inf"))).item(),
            torch.nextafter(torch.tensor(-2.0**-11), torch.tensor(float("-inf"))).item(),
        ],
        [
            0.75 * 2.0**-10,
            -0.75 * 2.0**-10,
            2.0**-10,
            -2.0**-10,
            1.5 * 2.0**-10,
            -1.5 * 2.0**-10,
            7.5 * 2.0**-10,
            -7.5 * 2.0**-10,
            2.0**-7,
            -2.0**-7,
        ],
        [1.0625, -1.0625, 1.1875, -1.1875],
        [
            120.0,
            -120.0,
            124.0,
            -124.0,
            128.0,
            -128.0,
            144.0,
            -144.0,
            160.0,
            -160.0,
            192.0,
            -192.0,
            224.0,
            -224.0,
            240.0,
            -240.0,
            256.0,
            -256.0,
            448.0,
            -448.0,
        ],
    ],
    ids=[
        "first-six",
        "zeros",
        "half-subnormal",
        "subnormal-boundaries",
        "rne-ties",
        "top-segment",
    ],
)
def test_deepseek_v4_fp8_pack_gfx942(values):
    if not is_hip_runtime():
        pytest.skip("This regression targets the ROCm software conversion path.")
    if "gfx942" not in torch.cuda.get_device_properties(0).gcnArchName:
        pytest.skip("This regression targets MI300X gfx942.")

    module = _probe_module()
    if len(values) % 2:
        values.append(0.0)
    input_tensor = torch.tensor(values, dtype=torch.float32, device="cuda")
    output_tensor = torch.empty_like(input_tensor, dtype=torch.uint8)
    metadata_tensor = torch.zeros(4, dtype=torch.float32, device="cuda")
    module.run(input_tensor, output_tensor, metadata_tensor)
    torch.cuda.synchronize()

    policy_max = metadata_tensor[0].item()
    assert policy_max == 224.0
    assert metadata_tensor[1].item() == 1.0
    assert metadata_tensor[2].item() == 0.0
    torch.testing.assert_close(
        output_tensor.cpu(),
        _expected_bytes(input_tensor, policy_max).cpu(),
        equal_nan=True,
    )


def test_deepseek_v4_fp8_pack_random_gfx942():
    if not is_hip_runtime():
        pytest.skip("This regression targets the ROCm software conversion path.")
    if "gfx942" not in torch.cuda.get_device_properties(0).gcnArchName:
        pytest.skip("This regression targets MI300X gfx942.")

    module = _probe_module()
    generator = torch.Generator(device="cuda")
    generator.manual_seed(12345)
    input_tensor = (torch.rand(4096, device="cuda", generator=generator) * 448 - 224)
    output_tensor = torch.empty_like(input_tensor, dtype=torch.uint8)
    metadata_tensor = torch.zeros(4, dtype=torch.float32, device="cuda")
    module.run(input_tensor, output_tensor, metadata_tensor)
    torch.cuda.synchronize()

    torch.testing.assert_close(
        output_tensor.cpu(),
        _expected_bytes(input_tensor, metadata_tensor[0].item()).cpu(),
        equal_nan=True,
    )


@pytest.mark.parametrize(
    "values",
    [
        [1.0, -1.0],
        [-1.0, 1.0],
        [128.0, 160.0],
        [160.0, 128.0],
        [224.0, -1e-8],
        [-1e-8, 224.0],
        [0.0, -0.0],
        [-0.0, 0.0],
        [2.0**-11, torch.nextafter(torch.tensor(2.0**-11), torch.tensor(0.0)).item()],
        [torch.nextafter(torch.tensor(2.0**-11), torch.tensor(0.0)).item(), 2.0**-11],
        [2.0**-11, torch.nextafter(torch.tensor(2.0**-11), torch.tensor(float("inf"))).item()],
        [torch.nextafter(torch.tensor(2.0**-11), torch.tensor(float("inf"))).item(), 2.0**-11],
        [-2.0**-11, torch.nextafter(torch.tensor(-2.0**-11), torch.tensor(0.0)).item()],
        [torch.nextafter(torch.tensor(-2.0**-11), torch.tensor(0.0)).item(), -2.0**-11],
        [-2.0**-11, torch.nextafter(torch.tensor(-2.0**-11), torch.tensor(float("-inf"))).item()],
        [torch.nextafter(torch.tensor(-2.0**-11), torch.tensor(float("-inf"))).item(), -2.0**-11],
        [1.0625, -1.0625],
        [-1.0625, 1.0625],
        [224.0, -224.0],
        [-224.0, 224.0],
    ],
    ids=[
        "unit",
        "unit-swapped",
        "top-asymmetric",
        "top-asymmetric-swapped",
        "first-six-tail",
        "first-six-tail-swapped",
        "signed-zero",
        "signed-zero-swapped",
        "positive-below",
        "positive-below-swapped",
        "positive-above",
        "positive-above-swapped",
        "negative-above-zero",
        "negative-above-zero-swapped",
        "negative-below",
        "negative-below-swapped",
        "rne-tie",
        "rne-tie-swapped",
        "policy-max",
        "policy-max-swapped",
    ],
)
def test_deepseek_v4_fp8_pack_asymmetric_lanes_gfx942(values):
    if not is_hip_runtime():
        pytest.skip("This regression targets the ROCm software conversion path.")
    if "gfx942" not in torch.cuda.get_device_properties(0).gcnArchName:
        pytest.skip("This regression targets MI300X gfx942.")

    module = _probe_module()
    input_tensor = torch.tensor(values, dtype=torch.float32, device="cuda")
    output_tensor = torch.empty_like(input_tensor, dtype=torch.uint8)
    metadata_tensor = torch.zeros(4, dtype=torch.float32, device="cuda")
    module.run(input_tensor, output_tensor, metadata_tensor)
    torch.cuda.synchronize()

    assert metadata_tensor[0].item() == 224.0
    assert metadata_tensor[1].item() == 1.0
    assert metadata_tensor[2].item() == 0.0
    expected = _expected_bytes(input_tensor, metadata_tensor[0].item())
    assert torch.equal(output_tensor, expected)
