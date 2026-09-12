from types import SimpleNamespace

import torch

import sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.cosmos3 as cosmos3
from sglang.multimodal_gen.configs.sample.sampling_params import DataType
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.cosmos3 import Cosmos3LatentPreparationStage


cosmos3.get_local_torch_device = lambda: torch.device("cpu")
vae = SimpleNamespace(config=SimpleNamespace(scale_factor_temporal=4, scale_factor_spatial=16))
transformer = SimpleNamespace(latent_channel=2)
stage = Cosmos3LatentPreparationStage.__new__(Cosmos3LatentPreparationStage)
stage.vae = vae
stage.transformer = transformer
stage.log_info = lambda *args, **kwargs: None
batch = SimpleNamespace(
    extra={}, num_frames=1, height=16, width=16,
    preprocessed_image=torch.zeros(1, 3, 16, 16),
    preprocessed_video=None, data_type=DataType.ACTION,
    generator=[torch.Generator().manual_seed(100), torch.Generator().manual_seed(101)],
    seed=100,
)
print("logical_effective_batch", 2)
print("conditioning_batch", batch.preprocessed_image.shape[0])
try:
    stage.forward(batch, SimpleNamespace())
except Exception as exc:
    print("stage_failure", type(exc).__name__, str(exc))
    raise
