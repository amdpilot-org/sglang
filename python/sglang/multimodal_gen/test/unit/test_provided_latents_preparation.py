from types import SimpleNamespace

import torch

from sglang.multimodal_gen.runtime.pipelines_core.stages.latent_preparation import (
    LatentPreparationStage,
)


class _PackingConfig:
    def __init__(self):
        self.pack_calls = 0

    def prepare_latent_shape(self, batch, batch_size, num_frames):
        return (batch_size, 4, 8, 8)

    def get_latent_dtype(self, dtype):
        return torch.float32

    def maybe_prepare_latent_ids(self, latents):
        return torch.zeros(latents.shape[0], 64, 4)

    def maybe_pack_latents(self, latents, batch_size, batch):
        self.pack_calls += 1
        return latents.reshape(batch_size, 4, 64).permute(0, 2, 1)


def _run(latents):
    stage = object.__new__(LatentPreparationStage)
    stage.scheduler = SimpleNamespace()
    stage.get_forward_latent_num_frames = lambda batch, server_args: 1
    batch = SimpleNamespace(
        batch_size=1,
        generator=None,
        latents=latents,
        height=16,
        width=16,
        num_frames=1,
        latent_ids=None,
        raw_latent_shape=None,
        num_outputs_per_prompt=1,
        prompt_embeds=[torch.zeros(1, 1)],
    )
    config = _PackingConfig()
    result = stage.forward(batch, SimpleNamespace(pipeline_config=config))
    return result, config


def test_unpacked_provided_latents_receive_ids_and_packing():
    result, config = _run(torch.zeros(1, 4, 8, 8))

    assert result.latent_ids.shape == (1, 64, 4)
    assert result.latents.shape == (1, 64, 4)
    assert config.pack_calls == 1


def test_already_packed_provided_latents_are_not_packed_twice():
    packed = torch.arange(64, dtype=torch.float32).reshape(1, 16, 4)
    result, config = _run(packed.clone())

    assert config.pack_calls == 0
    assert result.latent_ids is None
    assert torch.equal(result.latents.cpu(), packed)


def test_drawn_and_unpacked_provided_latents_have_matching_metadata_shapes():
    drawn, _ = _run(None)
    provided, _ = _run(torch.zeros(1, 4, 8, 8))

    assert drawn.latents.shape == provided.latents.shape
    assert drawn.latent_ids.shape == provided.latent_ids.shape
    assert drawn.raw_latent_shape == provided.raw_latent_shape
