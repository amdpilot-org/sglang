# SPDX-License-Identifier: Apache-2.0

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
