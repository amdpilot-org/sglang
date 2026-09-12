# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, patch

from sglang.multimodal_gen.runtime.models.vaes.autoencoder_dc import AutoencoderDC


def test_tiling_controls_are_forwarded_to_loaded_inner_model():
    vae = AutoencoderDC()
    vae._inner_model = Mock()

    vae.enable_tiling(
        tile_sample_min_height=512,
        tile_sample_min_width=768,
        tile_sample_stride_height=0.75,
        tile_sample_stride_width=0.5,
    )
    vae.disable_tiling()

    vae._inner_model.enable_tiling.assert_called_once_with(
        tile_sample_min_height=512,
        tile_sample_min_width=768,
        tile_sample_stride_height=0.75,
        tile_sample_stride_width=0.5,
    )
    vae._inner_model.disable_tiling.assert_called_once_with()


def test_enable_tiling_initializes_the_inner_model_before_forwarding():
    vae = AutoencoderDC()
    inner_model = Mock()

    with patch.object(
        vae,
        "_ensure_inner_model",
        side_effect=lambda: setattr(vae, "_inner_model", inner_model),
    ) as ensure_inner_model:
        vae.enable_tiling()

    ensure_inner_model.assert_called_once_with()
    inner_model.enable_tiling.assert_called_once_with(
        tile_sample_min_height=None,
        tile_sample_min_width=None,
        tile_sample_stride_height=None,
        tile_sample_stride_width=None,
    )


def test_disable_tiling_initializes_the_inner_model_before_forwarding():
    vae = AutoencoderDC()
    inner_model = Mock()

    with patch.object(
        vae,
        "_ensure_inner_model",
        side_effect=lambda: setattr(vae, "_inner_model", inner_model),
    ) as ensure_inner_model:
        vae.disable_tiling()

    ensure_inner_model.assert_called_once_with()
    inner_model.disable_tiling.assert_called_once_with()
