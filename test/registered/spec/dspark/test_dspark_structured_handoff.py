"""Structured-input parity for the reduced DSpark handoff primitives."""

import unittest

import torch

from sglang.kernels.ops.speculative.dspark.dspark_draft_model import CommitKvProj
from sglang.kernels.ops.speculative.dspark.dspark_verify_window import (
    BuildCommitInjectLayout,
)
from sglang.srt.speculative.dspark_components.dspark_draft import (
    select_draft_hidden_without_anchor,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")

DEVICE = torch.device("cuda")
BATCH_SIZE = 4
GAMMA = 5
HIDDEN_SIZE = 384
HEAD_DIM = 128
NUM_STAGES = 3
STRIDE = 6
NUM_POOL_REQUESTS = 8
POOL_LENGTH = 64
NUM_FULL_LOCATIONS = 512

CASE_NAMES = (
    "zeros",
    "tiny_signed",
    "mixed_magnitudes",
    "cancellation_pairs",
    "skewed_state",
    "boundary_lengths",
)


class _Bf16Linear(torch.nn.Module):
    quant_method = None

    def __init__(self, weight):
        super().__init__()
        self.weight = weight

    def forward(self, hidden):
        return torch.nn.functional.linear(hidden, self.weight), None


def _structured_hidden(case_name):
    rows = BATCH_SIZE * (GAMMA + 1)
    index = torch.arange(rows * HIDDEN_SIZE, dtype=torch.float32)
    if case_name == "zeros":
        values = torch.zeros_like(index)
    elif case_name == "tiny_signed":
        values = torch.where(index % 2 == 0, 1.0e-4, -1.0e-4)
    elif case_name == "mixed_magnitudes":
        values = torch.where(
            index % 3 == 0,
            (index % 7 + 1).float(),
            (index % 11 + 1) * 1.0e-3,
        )
    elif case_name == "cancellation_pairs":
        pair_magnitude = (index // 2) % 5 + 1
        values = torch.where(index % 2 == 0, pair_magnitude, -pair_magnitude)
    elif case_name == "skewed_state":
        values = torch.where(index % 10 == 0, 4.0, 1.0e-4)
    else:
        values = torch.where(index % 2 == 0, 1.0 / 256.0, -1.0 / 256.0)
    return values.reshape(rows, HIDDEN_SIZE).to(device=DEVICE, dtype=torch.bfloat16)


def _commit_lens(case_name):
    values = {
        "zeros": [0, 0, 0, 0],
        "tiny_signed": [0, 1, 2, 3],
        "mixed_magnitudes": [0, 2, 4, 6],
        "cancellation_pairs": [1, 1, 5, 5],
        "skewed_state": [0, 0, 0, 6],
        "boundary_lengths": [0, 1, 5, 6],
    }
    return torch.tensor(values[case_name], device=DEVICE, dtype=torch.int32)


@unittest.skipUnless(torch.cuda.is_available(), "Requires one CUDA/HIP GPU")
class DSparkStructuredHandoffTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        generator = torch.Generator(device="cpu").manual_seed(20260909)
        weights = []
        for _ in range(NUM_STAGES):
            weight = (
                torch.randn(
                    HEAD_DIM,
                    HIDDEN_SIZE,
                    device="cpu",
                    generator=generator,
                )
                * 0.01
            )
            weight[:, 0] = 0.25
            weight[:, 1] = -0.25
            weights.append(weight.to(device=DEVICE, dtype=torch.bfloat16))
        cls.projection_linears = [_Bf16Linear(weight) for weight in weights]

        cls.req_pool_indices = torch.tensor(
            [0, 2, 4, 6], device=DEVICE, dtype=torch.int32
        )
        cls.prefix_lens = torch.tensor(
            [7, 11, 13, 17], device=DEVICE, dtype=torch.int64
        )
        cls.block_pos_offsets = torch.arange(STRIDE, device=DEVICE)
        cls.req_to_token = torch.arange(
            NUM_POOL_REQUESTS * POOL_LENGTH, device=DEVICE, dtype=torch.int64
        ).reshape(NUM_POOL_REQUESTS, POOL_LENGTH)
        cls.full_to_swa_mapping = (
            torch.arange(NUM_FULL_LOCATIONS, device=DEVICE, dtype=torch.int64) * 3
            % 501
        )

    def _check_hidden_handoff(self, case_name):
        hidden_states = _structured_hidden(case_name)
        model_hidden, draft_hidden_3d = select_draft_hidden_without_anchor(
            hidden_states,
            bs=BATCH_SIZE,
            gamma=GAMMA,
        )

        hidden_by_query = hidden_states.reshape(
            BATCH_SIZE, GAMMA + 1, HIDDEN_SIZE
        )
        selected = hidden_by_query[:, 1:]
        model_hidden_reference = selected.reshape(BATCH_SIZE * GAMMA, HIDDEN_SIZE)
        draft_hidden_reference = selected.reshape(BATCH_SIZE, GAMMA, HIDDEN_SIZE)

        self.assertEqual(model_hidden.shape, model_hidden_reference.shape)
        self.assertEqual(model_hidden.dtype, model_hidden_reference.dtype)
        self.assertTrue(torch.equal(model_hidden, model_hidden_reference))
        self.assertEqual(draft_hidden_3d.shape, draft_hidden_reference.shape)
        self.assertEqual(draft_hidden_3d.dtype, draft_hidden_reference.dtype)
        self.assertTrue(torch.equal(draft_hidden_3d, draft_hidden_reference))
        return model_hidden

    def _check_projection_handoff(self, main_hidden):
        projected = CommitKvProj.triton(
            main_x=main_hidden,
            wkv_linears=self.projection_linears,
        )
        self.assertEqual(len(projected), NUM_STAGES)

        for stage_index, (got, linear) in enumerate(
            zip(projected, self.projection_linears)
        ):
            reference = torch.nn.functional.linear(
                main_hidden.detach().cpu().to(torch.float32),
                linear.weight.detach().cpu().to(torch.float32),
            ).to(torch.bfloat16)
            self.assertEqual(got.shape, reference.shape)
            self.assertEqual(got.dtype, torch.bfloat16)
            torch.testing.assert_close(
                got.float(),
                reference.to(DEVICE).float(),
                rtol=2.0e-2,
                atol=2.0e-3,
            )
            self.assertTrue(got.is_contiguous())

    def _check_commit_handoff(self, case_name):
        commit_lens = _commit_lens(case_name)
        got = BuildCommitInjectLayout.triton(
            req_pool_indices=self.req_pool_indices,
            req_to_token=self.req_to_token,
            prefix_lens=self.prefix_lens,
            block_pos_offsets=self.block_pos_offsets,
            full_to_swa_mapping=self.full_to_swa_mapping,
            commit_lens=commit_lens,
            stride=STRIDE,
        )

        positions_2d = (
            self.prefix_lens[:, None] + self.block_pos_offsets[None, :]
        )
        positions_reference = positions_2d.reshape(-1)
        cache_locations = self.req_to_token[
            self.req_pool_indices.long()[:, None], positions_2d
        ]
        swa_locations = self.full_to_swa_mapping[cache_locations]
        committed = (
            torch.arange(STRIDE, device=DEVICE)[None, :]
            < commit_lens[:, None]
        )
        swa_reference = torch.where(
            committed,
            swa_locations,
            torch.full_like(swa_locations, -1),
        ).reshape(-1).to(torch.int32)

        self.assertEqual(got.swa_loc.shape, swa_reference.shape)
        self.assertEqual(got.swa_loc.dtype, swa_reference.dtype)
        self.assertTrue(torch.equal(got.swa_loc, swa_reference))
        self.assertEqual(got.positions.shape, positions_reference.shape)
        self.assertEqual(got.positions.dtype, positions_reference.dtype)
        self.assertTrue(torch.equal(got.positions, positions_reference))

    def test_structured_handoff(self):
        for case_name in CASE_NAMES:
            with self.subTest(case_name=case_name):
                main_hidden = self._check_hidden_handoff(case_name)
                self._check_projection_handoff(main_hidden)
                self._check_commit_handoff(case_name)


if __name__ == "__main__":
    unittest.main()
