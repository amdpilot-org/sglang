from unittest.mock import patch

import pytest

from sglang.srt.models.mimo_audio import AudioEncoderAttention


@pytest.mark.parametrize("dp_attention_enabled", [False, True])
def test_audio_encoder_attention_uses_dp_attention_reduce(dp_attention_enabled):
    with (
        patch(
            "sglang.srt.models.mimo_audio.is_dp_attention_enabled",
            return_value=dp_attention_enabled,
        ),
        patch("sglang.srt.models.mimo_audio.VisionAttention") as vision_attention,
    ):
        AudioEncoderAttention(embed_dim=128, num_heads=4)

    assert vision_attention.call_args.kwargs["use_dp_attention_reduce"] is (
        dp_attention_enabled
    )
