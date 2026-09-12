"""Regression tests for multimodal mRoPE positions during draft extend."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

PROMPT_LEN = 16


def _mm_input(delta=5):
    return SimpleNamespace(
        mrope_positions=torch.arange(PROMPT_LEN).unsqueeze(0).repeat(3, 1),
        mrope_position_delta=torch.tensor([[delta]]),
        mrope_position_delta_repeated_cache=None,
    )


class TestMropeDraftExtendPositions(CustomTestCase):
    def setUp(self):
        override = get_context().override_server_args(model_path="dummy")
        override.install()
        self.addCleanup(override.restore)

    def _compute(self, multimodal_inputs, extend_lens, prefix_lens):
        forward_batch = ForwardBatch.__new__(ForwardBatch)
        forward_batch.forward_mode = ForwardMode.DRAFT_EXTEND_V2
        forward_batch.seq_lens_cpu = torch.tensor(prefix_lens, dtype=torch.int64)
        batch = SimpleNamespace(
            multimodal_inputs=multimodal_inputs,
            extend_lens=extend_lens,
            prefix_lens=prefix_lens,
        )
        forward_batch._compute_mrope_positions(SimpleNamespace(device="cpu"), batch)
        return forward_batch.mrope_positions

    def test_batched_draft_extend_emits_one_position_per_token(self):
        batch_size, num_draft_tokens, prefix_len = 10, 4, PROMPT_LEN + 20
        positions = self._compute(
            [_mm_input() for _ in range(batch_size)],
            [num_draft_tokens] * batch_size,
            [prefix_len] * batch_size,
        )

        self.assertEqual(positions.shape, (3, batch_size * num_draft_tokens))

    def test_ragged_batch_uses_each_request_delta_and_extend_length(self):
        positions = self._compute(
            [_mm_input(delta=3), _mm_input(delta=-2)],
            extend_lens=[2, 5],
            prefix_lens=[PROMPT_LEN + 4, PROMPT_LEN + 9],
        )
        expected = torch.cat(
            [
                torch.arange(PROMPT_LEN + 4, PROMPT_LEN + 6) + 3,
                torch.arange(PROMPT_LEN + 9, PROMPT_LEN + 14) - 2,
            ]
        )

        self.assertEqual(positions.shape, (3, 7))
        for axis in range(3):
            self.assertTrue(torch.equal(positions[axis], expected))

    def test_single_token_extend_matches_decode_position(self):
        prefix_len = PROMPT_LEN + 7
        mm_input = _mm_input()
        extend = self._compute([mm_input], [1], [prefix_len])
        decode = ForwardBatch.__new__(ForwardBatch)._expand_mrope_from_input(
            _mm_input(), prefix_len + 1
        )

        self.assertTrue(torch.equal(extend, decode))

    def test_in_range_prompt_slice_is_unchanged(self):
        prefix_len, extend_len = 8, 4
        positions = self._compute([_mm_input()], [extend_len], [prefix_len])
        expected = torch.arange(prefix_len, prefix_len + extend_len)

        self.assertEqual(positions.shape, (3, extend_len))
        for axis in range(3):
            self.assertTrue(torch.equal(positions[axis], expected))


if __name__ == "__main__":
    unittest.main()
