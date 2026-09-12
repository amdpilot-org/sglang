import types
import unittest
from unittest.mock import patch
import inspect

import torch

from sglang.srt.speculative.dspark_components.dspark_planner import (
    alloc_verify_window,
    clamp_verify_lens,
)
from sglang.srt.speculative.dspark_components.dspark_worker_v2 import DSparkWorkerV2
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDSparkContextBoundary(unittest.TestCase):
    def test_worker_uses_model_context_not_larger_allocation_width(self):
        source = inspect.getsource(DSparkWorkerV2._forward_decode)
        self.assertIn("self.model_runner.model_config.context_len", source)
        self.assertNotIn(
            "max_context_len = self.model_runner.req_to_token_pool.max_context_len",
            source,
        )

    def test_verify_lens_are_clamped_per_request(self):
        actual = clamp_verify_lens(
            requested_verify_lens=torch.tensor([6, 6, 6, 6]),
            seq_lens=torch.tensor([10, 98, 99, 97]),
            remaining_generation_tokens=torch.tensor([20, 20, 20, 2]),
            max_context_len=100,
        )
        self.assertEqual(actual.tolist(), [6, 2, 1, 2])

    def test_verify_lens_reject_exhausted_request(self):
        with self.assertRaisesRegex(RuntimeError, "no context or generation budget"):
            clamp_verify_lens(
                requested_verify_lens=torch.tensor([6, 6]),
                seq_lens=torch.tensor([100, 20]),
                remaining_generation_tokens=torch.tensor([20, 0]),
                max_context_len=100,
            )

    def test_verify_window_repeats_last_legal_position_in_padded_tail(self):
        batch = types.SimpleNamespace(
            seq_lens=torch.tensor([1_048_574, 10]),
            req_pool_indices=torch.tensor([0, 1]),
        )
        model_runner = types.SimpleNamespace(
            req_to_token_pool=types.SimpleNamespace(
                req_to_token=torch.empty((2, 1_048_600), dtype=torch.int64)
            )
        )
        cache_locs = torch.arange(12, dtype=torch.int64)
        with patch(
            "sglang.srt.speculative.dspark_components.dspark_planner."
            "assign_extend_cache_locs_func",
            return_value=cache_locs,
        ):
            window = alloc_verify_window(
                batch=batch,
                bs=2,
                device="cpu",
                verify_num_draft_tokens=6,
                block_pos_offsets=torch.arange(6),
                model_runner=model_runner,
                verify_lens=torch.tensor([2, 6]),
                max_context_len=1_048_576,
            )

        self.assertEqual(
            window.positions_2d.tolist(),
            [
                [1_048_574, 1_048_575, 1_048_575, 1_048_575, 1_048_575, 1_048_575],
                [10, 11, 12, 13, 14, 15],
            ],
        )
        self.assertLess(int(window.positions_2d.max()), 1_048_576)


if __name__ == "__main__":
    unittest.main()
