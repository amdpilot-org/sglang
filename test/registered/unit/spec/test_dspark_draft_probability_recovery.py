"""Regression coverage for DSPARK draft multinomial probability recovery."""

import unittest

import torch

from sglang.srt.speculative.dspark_components.dspark_draft import (
    _DRAFT_PROBS,
    _one_hot_token0,
)
from sglang.srt.utils.invariants import expect


class TestDSparkDraftProbabilityRecovery(unittest.TestCase):
    def test_unrecovered_nan_probabilities_fail_multinomial(self):
        probabilities = torch.full((1, 4), float("nan"))

        with self.assertRaisesRegex(RuntimeError, "probability tensor"):
            torch.multinomial(probabilities, num_samples=1)

    def test_non_finite_softmax_rows_recover_to_token_zero(self):
        logits = torch.tensor(
            [
                [float("-inf"), float("-inf"), float("-inf"), float("-inf")],
                [float("inf"), 0.0, -1.0, -2.0],
            ]
        )
        probabilities = torch.softmax(logits, dim=-1)

        recovered = expect(_DRAFT_PROBS, probabilities)

        torch.testing.assert_close(
            recovered,
            torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
        )
        torch.testing.assert_close(
            torch.multinomial(recovered, num_samples=1),
            torch.zeros((2, 1), dtype=torch.long),
        )

    def test_valid_rows_are_unchanged(self):
        probabilities = torch.tensor(
            [[0.1, 0.2, 0.3, 0.4], [1.0, 0.0, 0.0, 0.0]]
        )

        recovered = _one_hot_token0(probabilities)

        torch.testing.assert_close(recovered, probabilities)
        self.assertNotEqual(recovered.data_ptr(), probabilities.data_ptr())


if __name__ == "__main__":
    unittest.main()
