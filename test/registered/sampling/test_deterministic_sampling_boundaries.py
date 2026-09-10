import math
import unittest

import torch

from sglang.srt.layers.sampler import top_k_top_p_min_p_sampling_from_probs_torch
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=14, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd")


def cumulative_oracle(probs, top_k, top_p):
    order = sorted(range(len(probs)), key=lambda index: (-probs[index], index))
    support = []
    cumulative_mass = 0.0
    for rank, index in enumerate(order):
        if rank >= top_k or probs[index] == 0.0:
            continue
        mass_before = cumulative_mass
        if mass_before > top_p:
            break
        support.append(index)
        cumulative_mass = math.fsum((cumulative_mass, probs[index]))

    kept_mass = math.fsum(probs[index] for index in support)
    normalized = [probs[index] / kept_mass for index in support]
    return support, normalized


class TestDeterministicSamplingBoundaries(CustomTestCase):
    def _sample(self, probs, top_k, top_p):
        row_count = len(probs)
        return top_k_top_p_min_p_sampling_from_probs_torch(
            torch.tensor([probs], dtype=torch.float32, device="cuda"),
            torch.tensor([top_k], dtype=torch.int64, device="cuda"),
            torch.tensor([top_p], dtype=torch.float32, device="cuda"),
            torch.zeros(1, dtype=torch.float32, device="cuda"),
            False,
            torch.tensor([0x30344], dtype=torch.uint64, device="cuda"),
            torch.zeros(1, dtype=torch.int64, device="cuda"),
        ).item()

    def test_boundary_supports(self):
        cases = [
            ([0.0, 0.2, 0.3, 0.5, 0.0], 5, 1.0),
            ([1.0 - 2e-7, 5e-8, 5e-8, 5e-8, 5e-8], 5, 1.0),
            ([0.1, 0.2, 0.3, 0.4, 0.0, 0.0, 0.0, 0.0], 8, 1.0),
            ([0.5, 0.25, 0.125, 0.125], 4, 0.75),
            ([0.5, 0.25, 0.125, 0.125], 4, 0.7500001),
            ([0.5, 0.25, 0.125, 0.125], 2, 1.0),
        ]
        for probs, top_k, top_p in cases:
            with self.subTest(probs=probs, top_k=top_k, top_p=top_p):
                support, _ = cumulative_oracle(probs, top_k, top_p)
                self.assertIn(self._sample(probs, top_k, top_p), support)

    def test_finite_trial_distribution(self):
        probs = [0.125, 0.25, 0.25, 0.375]
        top_k = 4
        top_p = 0.625
        trial_count = 512
        support, normalized = cumulative_oracle(probs, top_k, top_p)
        sampled = top_k_top_p_min_p_sampling_from_probs_torch(
            torch.tensor([probs] * trial_count, dtype=torch.float32, device="cuda"),
            torch.full((trial_count,), top_k, dtype=torch.int64, device="cuda"),
            torch.full((trial_count,), top_p, dtype=torch.float32, device="cuda"),
            torch.zeros(trial_count, dtype=torch.float32, device="cuda"),
            False,
            torch.arange(trial_count, dtype=torch.int64, device="cuda").to(
                torch.uint64
            ),
            torch.zeros(trial_count, dtype=torch.int64, device="cuda"),
        )
        counts = torch.bincount(sampled, minlength=len(probs)).cpu()
        self.assertTrue(torch.isin(sampled, torch.tensor(support, device="cuda")).all())

        expected = [0.0] * len(probs)
        for token, probability in zip(support, normalized):
            expected[token] = trial_count * probability
        chi_square = sum(
            (observed.item() - expected_value) ** 2 / expected_value
            for observed, expected_value in zip(counts, expected)
            if expected_value > 0
        )
        self.assertLess(chi_square, 10.0)


if __name__ == "__main__":
    unittest.main()
