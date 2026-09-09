# SPDX-License-Identifier: Apache-2.0
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.configs.pipeline_configs.longlive2 import (
    LongLive2T2VConfig,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.longlive2 import (
    LongLive2CausalDenoisingStage,
    _causal_block_count,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="LongLive2 causal denoising requires a GPU"
)


def _make_stage(latents, calls):
    stage = LongLive2CausalDenoisingStage.__new__(LongLive2CausalDenoisingStage)
    stage.transformer = SimpleNamespace(independent_first_frame=False)
    stage.num_frames_per_block = 8
    stage.num_token_per_frame = 1
    stage.pos_start_base = 0
    stage.causal_kv_cache = object()
    stage.crossattn_cache = object()
    stage._i2v_image_latent = None
    stage._rope_temporal_offset = 0.0

    context = SimpleNamespace(
        target_dtype=torch.bfloat16,
        autocast_enabled=False,
        scheduler=object(),
        device=torch.device("cuda"),
        timesteps=torch.tensor([1000], device="cuda"),
        image_kwargs={},
        pos_cond_kwargs={},
        latents=latents,
        prompt_embeds=torch.zeros((1, 1), device="cuda"),
        num_frames=latents.shape[2],
        height=latents.shape[3],
        width=latents.shape[4],
    )

    def tiny_denoiser(
        batch,
        server_args,
        *,
        chunk_latents,
        start_frame,
        **kwargs,
    ):
        assert chunk_latents.is_cuda
        denoised = chunk_latents.mul(2).add_(1)
        torch.cuda.synchronize()
        calls.append(
            (
                int(chunk_latents.shape[2]),
                int(start_frame),
                tuple(denoised.shape),
            )
        )
        return denoised

    stage._prepare_causal_dmd_forward_context = lambda batch, server_args: context
    stage._get_max_text_len = lambda server_args: 1
    stage._causal_kv_cache_kwargs_for_batch = lambda batch: {}
    stage._cache_needs_reinit_for_batch = lambda cache, batch: False
    stage._reset_causal_caches = lambda **kwargs: None
    stage._validate_block_prompt_count = lambda batch, block_sizes: None
    stage._set_rope_temporal_offset = lambda batch, shot_index: None
    stage._shot_index = lambda batch, block_index: 0
    stage._is_scene_cut = lambda batch, block_index: False
    stage._select_block_prompt_embeds = (
        lambda batch, prompt_embeds, block_index: prompt_embeds
    )
    stage._select_block_cond_kwargs = (
        lambda batch, cond_kwargs, block_index: cond_kwargs
    )
    stage._reset_crossattn_cache_for_block = lambda batch, *caches: None
    stage._denoise_and_update_causal_block = tiny_denoiser
    stage.progress_bar = lambda total: nullcontext(
        SimpleNamespace(update=lambda: None)
    )
    return stage


def _make_batch(latents):
    return SimpleNamespace(
        latents=latents,
        image_latent=None,
        do_classifier_free_guidance=False,
        record_stage_iterations=lambda completed, total: None,
    )


@pytest.mark.parametrize(
    ("pixel_frames", "latent_frames", "block_count"),
    [(29, 8, 1), (61, 16, 2), (93, 24, 3)],
)
def test_valid_frame_counts_preserve_latent_length(
    pixel_frames,
    latent_frames,
    block_count,
):
    config = LongLive2T2VConfig()
    assert config.adjust_num_frames(pixel_frames, log_adjustment=False) == pixel_frames

    batch = SimpleNamespace(num_frames=pixel_frames)
    server_args = SimpleNamespace(pipeline_config=config)
    assert _causal_block_count(batch, server_args) == block_count

    calls = []
    latents = torch.arange(
        latent_frames * 4, dtype=torch.float32, device="cuda"
    ).reshape(1, 2, latent_frames, 2, 1)
    original_shape = latents.shape
    stage = _make_stage(latents, calls)
    result = stage.forward(_make_batch(latents), SimpleNamespace())

    assert result.latents.shape == original_shape
    block_shape = (1, 2, 8, 2, 1)
    assert calls == [
        (8, block_index * 8, block_shape)
        for block_index in range(block_count)
    ]


@pytest.mark.parametrize("pixel_frames", [65, 66])
def test_unadjusted_frame_counts_fail_before_denoising(pixel_frames):
    config = LongLive2T2VConfig()
    batch = SimpleNamespace(num_frames=pixel_frames)
    server_args = SimpleNamespace(pipeline_config=config)

    with pytest.raises(
        ValueError,
        match="LongLive2 latent frames must be divisible by num_frames_per_block",
    ):
        _causal_block_count(batch, server_args)


@pytest.mark.parametrize("latent_frames", [7, 9])
def test_invalid_latent_frames_fail_before_denoising(latent_frames):
    calls = []
    latents = torch.zeros(
        (1, 2, latent_frames, 2, 1), dtype=torch.float32, device="cuda"
    )
    stage = _make_stage(latents, calls)

    with pytest.raises(
        ValueError,
        match="num_frames must be divisible by num_frames_per_block",
    ):
        stage.forward(_make_batch(latents), SimpleNamespace())

    assert calls == []
