"""Focused regression evidence for upstream issue #34367.

This intentionally exercises configuration/request normalization only.  It does not
claim to run the weight-dependent LongLive2 denoiser.
"""

import pytest

from sglang.multimodal_gen.configs.pipeline_configs.longlive2 import (
    LongLive2T2VConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.wan import WanI2VCommonConfig
from sglang.multimodal_gen.test.server.testcase_configs import (
    LONGLIVE2_I2V_CI_sampling_params,
)


def _latent_frames(config: LongLive2T2VConfig, num_frames: int) -> int:
    temporal_scale = config.vae_config.arch_config.scale_factor_temporal
    return (num_frames - 1) // temporal_scale + 1


def test_reported_ci_request_reaches_a_complete_causal_block():
    config = LongLive2T2VConfig()
    requested = LONGLIVE2_I2V_CI_sampling_params.num_frames
    adjusted = config.adjust_num_frames(requested, log_adjustment=False)
    latent_frames = _latent_frames(config, adjusted)

    assert requested == 61
    assert adjusted == 61
    assert latent_frames == 16
    assert latent_frames % config.dit_config.arch_config.num_frames_per_block == 0


@pytest.mark.parametrize(
    ("requested", "expected_frames", "expected_latents"),
    [
        (17, 29, 8),  # below the first complete 8-latent-frame block
        (29, 29, 8),  # exact first-block boundary
        (60, 61, 16),  # Wan alignment first rounds to 61
        (62, 61, 16),  # Wan alignment rounds down to 61
        (65, 61, 16),  # incompatible 17-latent request rounds to two blocks
        (93, 93, 24),  # exact three-block boundary
    ],
)
def test_independent_frame_alignment_boundaries(
    requested: int, expected_frames: int, expected_latents: int
):
    config = LongLive2T2VConfig()
    adjusted = config.adjust_num_frames(requested, log_adjustment=False)

    assert adjusted == expected_frames
    assert _latent_frames(config, adjusted) == expected_latents
    assert expected_latents % config.dit_config.arch_config.num_frames_per_block == 0


def test_failing_before_reconstruction_without_longlive2_adjustment():
    config = LongLive2T2VConfig()

    # Reconstruct the faulty path by applying only the inherited Wan adjustment,
    # bypassing the LongLive2 model-specific block alignment.
    adjusted = WanI2VCommonConfig.adjust_num_frames(
        config, 17, log_adjustment=False
    )
    latent_frames = _latent_frames(config, adjusted)

    assert adjusted == 17
    assert latent_frames == 5
    assert latent_frames % config.dit_config.arch_config.num_frames_per_block != 0
