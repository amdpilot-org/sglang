# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.configs.sample.sampling_params import DataType
from sglang.multimodal_gen.runtime.candidate_trajectory import (
    ActionCandidateCapability,
    CandidateTrajectorySpec,
    reduce_action_candidates,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.cosmos3 import (
    Cosmos3DecodingStage,
    Cosmos3LatentPreparationStage,
)

CAPABILITY = ActionCandidateCapability(
    tensor="action_latents",
    reduction_order="after_denormalization",
    reducers=("none", "mean"),
    output_dtype="float32",
    output_shape="[H, D]",
    candidate_invariant_conditioning=("prompt",),
    auxiliary_branch_required=False,
)


@pytest.mark.parametrize(
    "value, message",
    [
        ({"count": 0, "reducer": "mean"}, "positive int"),
        ({"count": 2}, "requires a model-supported reducer"),
        ({"count": 2, "reducer": "median"}, "model-registered reducer"),
        ({"count": 2, "reducer": "mean", "seed_policy": "shared"}, "per_candidate"),
        ({"count": 2, "reducer": "mean", "surprise": True}, "Unknown"),
    ],
)
def test_candidate_spec_rejects_ambiguous_contracts(value, message):
    with pytest.raises(ValueError, match=message):
        CandidateTrajectorySpec.from_value(value)


def test_count_one_is_bitwise_identical():
    candidate = torch.randn(1, 4, 3)
    reduced = reduce_action_candidates(candidate, CandidateTrajectorySpec(), CAPABILITY)
    assert torch.equal(reduced, candidate[0])


def test_mean_matches_independent_reference_and_validates_candidate_axis():
    candidates = torch.tensor(
        [[[1.0, 2.0]], [[3.0, 6.0]], [[8.0, 10.0]]], dtype=torch.float32
    )
    spec = CandidateTrajectorySpec(count=3, reducer="mean")
    reduced = reduce_action_candidates(candidates, spec, CAPABILITY)
    reference = sum(candidates[i].double() for i in range(3)).div(3).float()
    torch.testing.assert_close(reduced, reference)

    with pytest.raises(RuntimeError, match="candidate axis"):
        reduce_action_candidates(candidates[:2], spec, CAPABILITY)


def test_cosmos3_returns_one_reduced_action_with_stable_candidate_identity():
    stage = Cosmos3DecodingStage.__new__(Cosmos3DecodingStage)
    stage.log_info = lambda *args, **kwargs: None
    candidates = torch.tensor(
        [[[1.0, 3.0]], [[5.0, 7.0]], [[9.0, 11.0]]], dtype=torch.float32
    )
    sampling_params = SimpleNamespace(
        candidate_trajectory=CandidateTrajectorySpec(
            count=3, reducer="mean", return_candidates=True
        ),
        action_stats_path=None,
        action_mode="policy",
    )
    batch = SimpleNamespace(
        action_latents=candidates,
        extra={"raw_action_dim": 2},
        sampling_params=sampling_params,
        data_type=DataType.ACTION,
        request_id="req-7",
        seed=100,
        num_inference_steps=4,
        num_frames=2,
        metrics=None,
    )

    result = stage.forward(batch, SimpleNamespace())
    payload = result.output[0]
    torch.testing.assert_close(torch.from_numpy(payload["actions"]), candidates.mean(0))
    assert payload["candidate_group"] == {
        "request_id": "req-7",
        "candidate_ids": [0, 1, 2],
        "reducer": "mean",
        "seed_policy": "per_candidate",
        "physical_batch_size": 3,
    }
    assert [item["candidate_id"] for item in payload["candidates"]] == [0, 1, 2]
    assert [item["seed"] for item in payload["candidates"]] == [100, 101, 102]
    assert result.action_pred.shape == (1, 1, 2)


def test_cosmos3_latents_use_candidate_batch_and_per_candidate_generators(monkeypatch):
    monkeypatch.setattr(
        "sglang.multimodal_gen.runtime.pipelines_core.stages."
        "model_specific_stages.cosmos3.get_local_torch_device",
        lambda: torch.device("cpu"),
    )

    class FakeVAE(torch.nn.Module):
        config = SimpleNamespace(
            scale_factor_temporal=4,
            scale_factor_spatial=16,
            latents_mean=[0.0, 0.0],
            latents_std=[1.0, 1.0],
        )

        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))

        def encode(self, video):
            latent = torch.zeros(video.shape[0], 2, 1, 1, 1)
            return SimpleNamespace(mode=lambda: latent)

    stage = Cosmos3LatentPreparationStage(FakeVAE(), SimpleNamespace(latent_channel=2))
    stage.log_info = lambda *args, **kwargs: None
    stage.use_declared_component = lambda **kwargs: nullcontext()
    batch = SimpleNamespace(
        extra={},
        num_frames=5,
        height=16,
        width=16,
        preprocessed_image=torch.zeros(1, 3, 16, 16),
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

    assert result.latents.shape == (2, 2, 2, 1, 1)
    assert result.extra["condition_latents"].shape[0] == 2
    reference = torch.cat(
        [
            torch.randn(
                (1, 2, 2, 1, 1),
                generator=torch.Generator().manual_seed(seed),
                dtype=torch.bfloat16,
            )
            for seed in (100, 101)
        ]
    )
    torch.testing.assert_close(result.latents[:, :, 1:], reference[:, :, 1:])


def test_cosmos3_latents_reject_generator_count_mismatch(monkeypatch):
    monkeypatch.setattr(
        "sglang.multimodal_gen.runtime.pipelines_core.stages."
        "model_specific_stages.cosmos3.get_local_torch_device",
        lambda: torch.device("cpu"),
    )
    stage = Cosmos3LatentPreparationStage(
        SimpleNamespace(
            config=SimpleNamespace(scale_factor_temporal=4, scale_factor_spatial=16)
        ),
        SimpleNamespace(latent_channel=2),
    )
    stage.log_info = lambda *args, **kwargs: None
    batch = SimpleNamespace(
        extra={},
        num_frames=1,
        height=16,
        width=16,
        preprocessed_image=None,
        preprocessed_video=None,
        data_type=DataType.ACTION,
        generator=[torch.Generator().manual_seed(100)],
        seed=100,
        batch_size=2,
    )

    with pytest.raises(ValueError, match="effective batch size of 2"):
        stage.forward(batch, SimpleNamespace())
