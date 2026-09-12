from types import SimpleNamespace

import torch

import sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.cosmos3 as cosmos3
from sglang.multimodal_gen.configs.sample.sampling_params import DataType
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.cosmos3 import (
    Cosmos3LatentPreparationStage,
)


device = torch.device("cuda")
cosmos3.get_local_torch_device = lambda: device
stage = Cosmos3LatentPreparationStage.__new__(Cosmos3LatentPreparationStage)
stage.vae = SimpleNamespace(config=SimpleNamespace(scale_factor_temporal=4, scale_factor_spatial=16))
stage.transformer = SimpleNamespace(latent_channel=2)
stage.log_info = lambda *args, **kwargs: None
seeds = (100, 101)
batch = SimpleNamespace(
    extra={},
    num_frames=5,
    height=16,
    width=16,
    preprocessed_image=None,
    preprocessed_video=None,
    data_type=DataType.ACTION,
    generator=[torch.Generator(device=device).manual_seed(seed) for seed in seeds],
    seed=seeds[0],
    batch_size=len(seeds),
    sampling_params=SimpleNamespace(action_mode=None),
    sound_duration=0.0,
)
actual = stage.forward(batch, SimpleNamespace()).latents
reference = torch.cat(
    [
        torch.randn(
            (1, 2, 2, 1, 1),
            generator=torch.Generator(device=device).manual_seed(seed),
            device=device,
            dtype=torch.bfloat16,
        )
        for seed in seeds
    ]
)
torch.testing.assert_close(actual, reference, rtol=0, atol=0)
print("device", torch.cuda.get_device_name())
print("physical_latent_batch", actual.shape[0])
print("sequential_parity", torch.equal(actual, reference))
print("candidate_distinct", not torch.equal(actual[0], actual[1]))

