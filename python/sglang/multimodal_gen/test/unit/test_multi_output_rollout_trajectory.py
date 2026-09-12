import torch

from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import (
    DiffGenerator,
)
from sglang.multimodal_gen.runtime.managers.gpu_worker import GPUWorker
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch
from sglang.multimodal_gen.runtime.post_training.rl_dataclasses import (
    RolloutDitTrajectory,
    RolloutTrajectoryData,
)


def _trajectory(tag: float) -> RolloutTrajectoryData:
    return RolloutTrajectoryData(
        rollout_log_probs=torch.full((1, 4), tag),
        dit_trajectory=RolloutDitTrajectory(
            latents=torch.full((1, 5, 2), tag),
            timesteps=torch.arange(4),
            sigmas=torch.linspace(1, 0, 5),
        ),
    )


def test_merge_and_result_keep_each_output_trajectory():
    merged = GPUWorker._merge_expanded_output_batches(
        [OutputBatch(rollout_trajectory_data=_trajectory(tag)) for tag in (1, 2, 3)]
    )

    assert merged.rollout_trajectory_data.rollout_log_probs.shape == (3, 4)
    results = [
        DiffGenerator._result_common(
            type(
                "Req",
                (),
                {
                    "data_type": None,
                    "height": 8,
                    "width": 8,
                    "num_frames": 1,
                    "prompt": "test",
                },
            )(),
            merged,
            0.0,
            output_index,
        )["rollout_trajectory_data"]
        for output_index in range(3)
    ]
    assert [result.rollout_log_probs[0, 0].item() for result in results] == [1, 2, 3]
    assert all(result.rollout_log_probs.shape == (1, 4) for result in results)


def test_merge_drops_incomplete_trajectory_group():
    merged = GPUWorker._merge_expanded_output_batches(
        [
            OutputBatch(rollout_trajectory_data=_trajectory(1)),
            OutputBatch(rollout_trajectory_data=None),
            OutputBatch(rollout_trajectory_data=_trajectory(3)),
        ]
    )

    assert merged.rollout_trajectory_data is None


def test_single_output_trajectory_is_preserved_by_identity():
    trajectory = _trajectory(1)
    merged = GPUWorker._merge_expanded_output_batches(
        [OutputBatch(rollout_trajectory_data=trajectory)]
    )

    assert merged.rollout_trajectory_data is trajectory
