import unittest

from sglang.srt.managers.tokenizer_manager_score_mixin import (
    TokenizerManagerScoreMixin,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _ScoreResultMixin(TokenizerManagerScoreMixin):
    def __init__(self):
        self.is_generation = False


class TestSequenceClassificationScoreResults(CustomTestCase):
    def setUp(self):
        self.mixin = _ScoreResultMixin()

    def test_scalar_logits_are_returned_as_single_label_rows(self):
        result = self.mixin._process_single_item_scoring_results(
            [
                {"embedding": 5.875, "meta_info": {"prompt_tokens": 4}},
                {"embedding": -10.109375, "meta_info": {"prompt_tokens": 6}},
            ],
            label_token_ids=None,
            apply_softmax=False,
        )

        self.assertEqual(result.scores, [[5.875], [-10.109375]])
        self.assertEqual(result.prompt_tokens, 10)

    def test_scalar_softmax_preserves_row_shape(self):
        result = self.mixin._process_single_item_scoring_results(
            [{"embedding": 3.25}],
            label_token_ids=None,
            apply_softmax=True,
        )

        self.assertEqual(result.scores, [[1.0]])

    def test_multi_label_logits_remain_vectors(self):
        result = self.mixin._process_single_item_scoring_results(
            [{"embedding": [2.0, -1.0]}],
            label_token_ids=None,
            apply_softmax=False,
        )

        self.assertEqual(result.scores, [[2.0, -1.0]])


if __name__ == "__main__":
    unittest.main(verbosity=3)
