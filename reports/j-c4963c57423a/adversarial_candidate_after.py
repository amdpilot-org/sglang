from types import SimpleNamespace

import torch

import sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.cosmos3 as cosmos3
from sglang.multimodal_gen.configs.sample.sampling_params import DataType
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.cosmos3 import (
    Cosmos3LatentPreparationStage,
)


cosmos3.get_local_torch_device = lambda: torch.device("cpu")
vae = SimpleNamespace(config=SimpleNamespace(scale_factor_temporal=4, scale_factor_spatial=16))
transformer = SimpleNamespace(latent_channel=2)
stage = Cosmos3LatentPreparationStage.__new__(Cosmos3LatentPreparationStage)
stage.vae = vae
stage.transformer = transformer
stage.log_info = lambda *args, **kwargs: None
batch = SimpleNamespace(
    extra={},
    num_frames=1,
    height=16,
    width=16,
    preprocessed_image=None,
    preprocessed_video=None,
    data_type=DataType.ACTION,
    generator=[
        torch.Generator().manual_seed(100),
        torch.Generator().manual_seed(101),
    ],
    seed=100,
    batch_size=2,
    sampling_params=SimpleNamespace(action_mode=None),
    sound_duration=0.0,
)
result = stage.forward(batch, SimpleNamespace())
print("logical_effective_batch", batch.batch_size)
print("physical_latent_batch", result.latents.shape[0])
print("latent_shape", tuple(result.latents.shape))

