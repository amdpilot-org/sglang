import unittest

from sglang.srt.arg_groups.validation_hook import validate_ngram_capacity
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestNgramServerArgs(CustomTestCase):
    def _make_args(self, **kwargs):
        args = ServerArgs(model_path="dummy", served_model_name="dummy", **kwargs)
        args.resolve_once()
        return args

    def test_capacity_equal_to_max_depth_is_rejected(self):
        args = self._make_args(
            speculative_algorithm="NGRAM",
            speculative_ngram_capacity=4,
            speculative_ngram_max_trie_depth=4,
        )
        with self.assertRaisesRegex(
            AssertionError,
            "speculative_ngram_capacity must be greater than speculative_ngram_max_trie_depth",
        ):
            validate_ngram_capacity(args)

    def test_capacity_below_max_depth_is_rejected(self):
        args = self._make_args(
            speculative_algorithm="NGRAM",
            speculative_ngram_capacity=3,
            speculative_ngram_max_trie_depth=4,
        )
        with self.assertRaisesRegex(
            AssertionError,
            "speculative_ngram_capacity must be greater than speculative_ngram_max_trie_depth",
        ):
            validate_ngram_capacity(args)

    def test_capacity_one_above_max_depth_is_accepted(self):
        args = self._make_args(
            speculative_algorithm="NGRAM",
            speculative_ngram_capacity=5,
            speculative_ngram_max_trie_depth=4,
        )
        validate_ngram_capacity(args)

    def test_non_ngram_algorithm_is_unaffected(self):
        args = self._make_args(
            speculative_algorithm="EAGLE",
            speculative_ngram_capacity=4,
            speculative_ngram_max_trie_depth=4,
        )
        validate_ngram_capacity(args)


if __name__ == "__main__":
    unittest.main()
