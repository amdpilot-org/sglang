import torch

from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import DiffGenerator
from sglang.multimodal_gen.runtime.managers.gpu_worker import GPUWorker
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch
from sglang.multimodal_gen.runtime.post_training.rl_dataclasses import (
    RolloutDenoisingEnv,
    RolloutTrajectoryData,
)


def _trajectory(tag: float) -> RolloutTrajectoryData:
    return RolloutTrajectoryData(
        rollout_log_probs=torch.full((1, 2), tag),
        denoising_env=RolloutDenoisingEnv(
            guidance=torch.tensor([tag]),
            image_kwargs={"pixels": torch.full((1, 2), tag)},
        ),
    )


def test_distinct_denoising_environments_follow_their_outputs():
    merged = GPUWorker._merge_expanded_output_batches(
        [OutputBatch(rollout_trajectory_data=_trajectory(tag)) for tag in (1.0, 2.0, 3.0)]
    )
    req = type(
        "Req",
        (),
        {"data_type": None, "height": 8, "width": 8, "num_frames": 1, "prompt": "x"},
    )()
    results = [
        DiffGenerator._result_common(req, merged, 0.0, i)["rollout_trajectory_data"]
        for i in range(3)
    ]
    assert [r.denoising_env.guidance.item() for r in results] == [1.0, 2.0, 3.0]
    assert [r.denoising_env.image_kwargs["pixels"][0, 0].item() for r in results] == [1.0, 2.0, 3.0]
