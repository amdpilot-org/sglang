import unittest

import torch

from sglang.srt.layers.sampler import (
    multinomial_with_seed,
    top_k_top_p_min_p_sampling_from_probs_torch,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA-compatible GPU is required")
class TestSeededMinPSampling(unittest.TestCase):
    def _sample(self, probs, min_ps, *, top_ks=None, positions=None):
        batch_size, vocab_size = probs.shape
        device = probs.device
        if top_ks is None:
            top_ks = torch.full(
                (batch_size,), vocab_size, dtype=torch.int32, device=device
            )
        if positions is None:
            positions = torch.full((batch_size,), 17, dtype=torch.int64, device=device)
        return top_k_top_p_min_p_sampling_from_probs_torch(
            probs=probs,
            top_ks=top_ks,
            top_ps=torch.ones(batch_size, device=device),
            min_ps=min_ps,
            need_min_p_sampling=True,
            sampling_seed=torch.arange(batch_size, dtype=torch.int64, device=device),
            positions=positions,
        )

    def test_seeded_min_p_is_deterministic_and_preserves_token_ids(self):
        batch_size = 128
        probs = torch.tensor([[0.05, 0.55, 0.10, 0.30]], device="cuda").repeat(
            batch_size, 1
        )
        min_ps = torch.full((batch_size,), 0.5, device="cuda")

        first = self._sample(probs, min_ps)
        second = self._sample(probs, min_ps)

        torch.testing.assert_close(first, second)
        self.assertTrue(torch.isin(first, torch.tensor([1, 3], device="cuda")).all())
        self.assertTrue((first == 1).any())
        self.assertTrue((first == 3).any())

    def test_seeded_min_p_matches_independently_normalized_weights(self):
        probs = torch.tensor(
            [[0.05, 0.55, 0.10, 0.30], [0.40, 0.10, 0.20, 0.30]],
            device="cuda",
        )
        seeds = torch.arange(2, dtype=torch.int64, device="cuda")
        positions = torch.tensor([17, 29], dtype=torch.int64, device="cuda")
        min_ps = torch.tensor([0.5, 0.75], device="cuda")

        actual = self._sample(probs, min_ps, positions=positions)

        sorted_probs, sorted_token_ids = probs.sort(dim=-1, descending=True)
        thresholds = sorted_probs[:, :1] * min_ps[:, None]
        filtered = sorted_probs.masked_fill(sorted_probs < thresholds, 0.0)
        normalized = filtered / filtered.sum(dim=-1, keepdim=True)
        sorted_samples = multinomial_with_seed(
            normalized.double().log(), seeds, positions
        )
        expected = sorted_token_ids.gather(1, sorted_samples).view(-1).to(torch.int32)

        torch.testing.assert_close(actual, expected)

    def test_threshold_is_inclusive_and_top_k_still_applies(self):
        batch_size = 64
        probs = torch.tensor([[0.4, 0.4, 0.2, 0.0]], device="cuda").repeat(
            batch_size, 1
        )
        min_ps = torch.ones(batch_size, device="cuda")
        top_ks = torch.full((batch_size,), 2, dtype=torch.int32, device="cuda")

        samples = self._sample(probs, min_ps, top_ks=top_ks)

        self.assertTrue(torch.isin(samples, torch.tensor([0, 1], device="cuda")).all())
        self.assertTrue((samples == 0).any())
        self.assertTrue((samples == 1).any())


if __name__ == "__main__":
    unittest.main()
