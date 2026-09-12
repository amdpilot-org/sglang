import unittest

import torch

from sglang.srt.utils.weight_versions import WeightVersionSpan
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.io_struct import (
    BatchEmbeddingOutput,
    BatchStrOutput,
    unwrap_from_pickle,
    wrap_as_pickle,
)
from sglang.srt.managers.multi_tokenizer_mixin import (
    TokenizerWorker,
    _handle_output_by_index,
    get_tokenizer_worker_class,
)

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


class CustomTokenizerWorker(TokenizerWorker):
    pass


class NotAWorker:
    pass


class DefaultServerArgs:
    def get_tokenizer_worker_class(self):
        return TokenizerWorker


class CustomServerArgs:
    def get_tokenizer_worker_class(self):
        return CustomTokenizerWorker


class InvalidServerArgs:
    def get_tokenizer_worker_class(self):
        return NotAWorker


def _make_batch_str_output() -> BatchStrOutput:
    return BatchStrOutput(
        rids=["rid-0", "rid-1"],
        spec_verify_ct=[0, 0],
        spec_num_correct_drafts=[0, 0],
        spec_correct_drafts_histogram=[[], []],
        finished_reasons=[None, {"type": "length"}],
        output_strs=["first", "second"],
        output_ids=[[1], [2]],
        prompt_tokens=[10, 20],
        completion_tokens=[1, 2],
        reasoning_tokens=[0, 0],
        cached_tokens=[3, 4],
        cached_tokens_details=[
            {"device": 3, "host": 0},
            {"device": 1, "host": 3},
        ],
        input_token_logprobs_val=[[], []],
        input_token_logprobs_idx=[[], []],
        output_token_logprobs_val=[[], []],
        output_token_logprobs_idx=[[], []],
        input_top_logprobs_val=[[], []],
        input_top_logprobs_idx=[[], []],
        output_top_logprobs_val=[[], []],
        output_top_logprobs_idx=[[], []],
        input_token_ids_logprobs_val=[[], []],
        input_token_ids_logprobs_idx=[[], []],
        output_token_ids_logprobs_val=[[], []],
        output_token_ids_logprobs_idx=[[], []],
        output_token_entropy_val=[0.0, 0.0],
        output_token_sampling_mask=[[], []],
        output_token_sampling_logprobs=[[], []],
        output_hidden_states=[None, None],
        routed_experts=[None, None],
        indexer_topk=[None, None],
        placeholder_tokens_idx=[None, None],
        placeholder_tokens_val=[None, None],
        retraction_counts=[0, 0],
        weight_versions=[
            [
                WeightVersionSpan(version="v1", start=0, end=3),
                WeightVersionSpan(version="v2", start=3, end=5),
            ],
            [WeightVersionSpan(version="v2", start=0, end=2)],
        ],
    )


def _make_batch_embedding_output(**overrides) -> BatchEmbeddingOutput:
    fields = dict(
        rids=["rid-0", "rid-1"],
        finished_reasons=[None, {"type": "length"}],
        embeddings=[[0.1, 0.2], [0.3, 0.4]],
        prompt_tokens=[10, 20],
        cached_tokens=[3, 4],
        placeholder_tokens_idx=None,
        placeholder_tokens_val=None,
        retraction_counts=[0, 2],
        cached_tokens_details=[
            {"device": 3, "host": 0},
            {"device": 1, "host": 3},
        ],
        time_stats=wrap_as_pickle(["stats-0", "stats-1"]),
        pooled_hidden_states=[torch.tensor([1.0]), torch.tensor([2.0])],
    )
    fields.update(overrides)
    return BatchEmbeddingOutput(**fields)


class TestMultiTokenizerMixin(unittest.TestCase):
    def test_batch_embedding_output_preserves_per_request_fields(self):
        single_output = _handle_output_by_index(_make_batch_embedding_output(), 1)

        self.assertEqual(single_output.rids, ["rid-1"])
        self.assertEqual(single_output.retraction_counts, [2])
        self.assertEqual(
            single_output.cached_tokens_details,
            [{"device": 1, "host": 3}],
        )
        self.assertEqual(unwrap_from_pickle(single_output.time_stats), ["stats-1"])
        torch.testing.assert_close(
            single_output.pooled_hidden_states[0], torch.tensor([2.0])
        )

    def test_batch_embedding_output_splits_stacked_pooled_hidden_states(self):
        output = _make_batch_embedding_output(
            pooled_hidden_states=[torch.tensor([[1.0, 1.5], [2.0, 2.5]])]
        )

        single_output = _handle_output_by_index(output, 1)

        self.assertEqual(len(single_output.pooled_hidden_states), 1)
        torch.testing.assert_close(
            single_output.pooled_hidden_states[0], torch.tensor([2.0, 2.5])
        )

    def test_batch_embedding_output_preserves_absent_optional_fields(self):
        output = _make_batch_embedding_output(
            retraction_counts=None,
            cached_tokens_details=None,
            time_stats=None,
            pooled_hidden_states=None,
        )

        single_output = _handle_output_by_index(output, 0)

        self.assertIsNone(single_output.retraction_counts)
        self.assertIsNone(single_output.cached_tokens_details)
        self.assertIsNone(single_output.time_stats)
        self.assertIsNone(single_output.pooled_hidden_states)

    def test_batch_str_output_preserves_cached_tokens_details(self):
        output = _make_batch_str_output()

        single_output = _handle_output_by_index(output, 1)

        self.assertEqual(single_output.rids, ["rid-1"])
        self.assertEqual(single_output.cached_tokens, [4])
        self.assertEqual(
            single_output.cached_tokens_details,
            [{"device": 1, "host": 3}],
        )

    def test_batch_str_output_keeps_weight_versions_nested_per_request(self):
        """Per-request segment lists stay one level nested after the split."""
        output = _make_batch_str_output()

        self.assertEqual(
            _handle_output_by_index(output, 0).weight_versions,
            [
                [
                    WeightVersionSpan(version="v1", start=0, end=3),
                    WeightVersionSpan(version="v2", start=3, end=5),
                ]
            ],
        )
        self.assertEqual(
            _handle_output_by_index(output, 1).weight_versions,
            [[WeightVersionSpan(version="v2", start=0, end=2)]],
        )

    def test_batch_str_output_without_weight_versions_stays_none(self):
        """An output from an older server without the field splits into None."""
        output = _make_batch_str_output()
        output.weight_versions = None

        self.assertIsNone(_handle_output_by_index(output, 0).weight_versions)

    def test_get_tokenizer_worker_class_uses_default(self):
        self.assertIs(get_tokenizer_worker_class(DefaultServerArgs()), TokenizerWorker)

    def test_get_tokenizer_worker_class_resolves_custom_class(self):
        self.assertIs(
            get_tokenizer_worker_class(CustomServerArgs()),
            CustomTokenizerWorker,
        )

    def test_get_tokenizer_worker_class_rejects_non_worker(self):
        with self.assertRaisesRegex(TypeError, "TokenizerWorker"):
            get_tokenizer_worker_class(InvalidServerArgs())


if __name__ == "__main__":
    unittest.main()
