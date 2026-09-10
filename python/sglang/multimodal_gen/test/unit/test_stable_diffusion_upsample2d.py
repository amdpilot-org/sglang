from __future__ import annotations

import pytest
import torch

from sglang.multimodal_gen.runtime.models.dits.stable_diffusion import Upsample2D


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Stable Diffusion upsample parity requires CUDA",
)


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "source_spatial,output_size,expected_spatial",
    [
        ((8, 11), None, (16, 22)),
        ((8, 11), (13, 17), (13, 17)),
        ((8, 11), (1, 1), (1, 1)),
        ((8, 11), (3, 5), (3, 5)),
        ((8, 11), (17, 23), (17, 23)),
    ],
)
def test_upsample2d_nearest_resize_matches_cpu_reference(
    dtype, source_spatial, output_size, expected_spatial
):
    torch.manual_seed(23494)
    channels = 3
    source = torch.randn(
        1,
        channels,
        *source_spatial,
        device="cpu",
        dtype=dtype,
    )
    module = Upsample2D(channels=channels).to(device="cuda", dtype=dtype).eval()
    module.conv = torch.nn.Identity()

    with torch.no_grad():
        actual = module(
            source.to(device="cuda"),
            output_size=output_size,
        )
        if output_size is None:
            expected = torch.nn.functional.interpolate(
                source,
                scale_factor=2.0,
                mode="nearest",
            )
        else:
            expected = torch.nn.functional.interpolate(
                source,
                size=output_size,
                mode="nearest",
            )

    assert actual.shape == (1, channels, *expected_spatial)
    assert actual.dtype == dtype
    torch.testing.assert_close(
        actual.cpu(),
        expected,
        rtol=0.0,
        atol=0.0,
    )
