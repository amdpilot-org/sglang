from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


def _processor():
    processor = LogitsProcessor.__new__(LogitsProcessor)
    processor.do_tensor_parallel_all_gather = True
    processor.do_tensor_parallel_all_gather_dp_attn = False
    processor.use_attn_tp_group = False
    processor.use_tp_lm_head_all_to_all = False
    return processor


def _metadata(**overrides):
    values = dict(
        forward_mode=ForwardMode.EXTEND,
        return_logprob=False,
        extend_return_logprob=False,
        is_prefill_only=False,
        is_speculative=False,
        can_run_decode_cuda_graph=False,
    )
    values.update(overrides)
    return LogitsMetadata(**values)


def test_tp_logits_gather_selected_for_ordinary_generation():
    assert _processor()._use_tp_logits_gather(_metadata())


def test_tp_logits_gather_is_disabled_during_graph_capture():
    flags = SimpleNamespace(capture=SimpleNamespace(disable_dispose_tensor=True))
    with patch("sglang.srt.layers.logits_processor.get_flags", return_value=flags):
        assert not _processor()._use_tp_logits_gather(_metadata())


@pytest.mark.parametrize(
    "override",
    [
        {"return_logprob": True},
        {"extend_return_logprob": True},
        {"is_prefill_only": True},
        {"is_speculative": True},
        {"forward_mode": ForwardMode.DLLM_EXTEND},
        {"can_run_decode_cuda_graph": True},
    ],
)
def test_tp_logits_gather_preserves_all_gather_for_incompatible_batches(override):
    assert not _processor()._use_tp_logits_gather(_metadata(**override))


class _FakeGroup:
    def __init__(self, rank, root_tokens):
        self.rank_in_group = rank
        self.root_tokens = root_tokens
        self.broadcast_calls = 0

    def broadcast(self, tensor, src=0):
        self.broadcast_calls += 1
        if self.rank_in_group != src:
            tensor.copy_(self.root_tokens)
        return tensor


def _runner(sampled_tokens):
    runner = SimpleNamespace()
    runner.sampling_observer = None
    runner._preprocess_logits = Mock(return_value=None)
    runner.sampler = Mock(return_value=sampled_tokens.clone())
    runner.ngram_embedding_manager = SimpleNamespace(update_after_decode=Mock())
    return runner


def _forward_batch():
    return SimpleNamespace(
        sampling_info=SimpleNamespace(grammar_mask=None),
        return_logprob=False,
        top_logprobs_nums=[],
        token_ids_logprobs=[],
        positions=torch.tensor([0, 0]),
        seq_lens=torch.tensor([1, 1]),
        input_ids=torch.tensor([1, 2]),
        forward_mode=ForwardMode.EXTEND,
    )


def test_tp_root_samples_once_then_broadcasts_tokens():
    expected = torch.tensor([7, 11])
    runner = _runner(expected)
    group = _FakeGroup(rank=0, root_tokens=expected)
    output = SimpleNamespace(
        tp_logits_gathered=True,
        next_token_logits=torch.randn(2, 16),
        auxiliary_device_output=None,
    )

    with patch(
        "sglang.srt.model_executor.model_runner.get_tp_group", return_value=group
    ):
        actual = ModelRunner.sample(runner, output, _forward_batch())

    torch.testing.assert_close(actual, expected)
    runner.sampler.assert_called_once()
    assert group.broadcast_calls == 1


def test_non_root_never_reads_missing_logits_and_receives_root_tokens():
    expected = torch.tensor([7, 11])
    runner = _runner(expected)
    group = _FakeGroup(rank=1, root_tokens=expected)
    output = SimpleNamespace(
        tp_logits_gathered=True,
        next_token_logits=None,
        auxiliary_device_output=None,
    )

    with patch(
        "sglang.srt.model_executor.model_runner.get_tp_group", return_value=group
    ):
        actual = ModelRunner.sample(runner, output, _forward_batch())

    torch.testing.assert_close(actual, expected)
    runner.sampler.assert_not_called()
    assert group.broadcast_calls == 1
    runner.ngram_embedding_manager.update_after_decode.assert_called_once()


def test_non_root_releases_rank_local_sampling_state():
    expected = torch.tensor([7, 11])
    runner = _runner(expected)
    group = _FakeGroup(rank=1, root_tokens=expected)
    forward_batch = _forward_batch()
    forward_batch.sampling_info = SimpleNamespace(grammar_mask=torch.ones(16))
    stale_auxiliary_output = object()
    output = SimpleNamespace(
        tp_logits_gathered=True,
        next_token_logits=None,
        auxiliary_device_output=stale_auxiliary_output,
    )

    with patch(
        "sglang.srt.model_executor.model_runner.get_tp_group", return_value=group
    ):
        ModelRunner.sample(runner, output, forward_batch)

    assert forward_batch.sampling_info.grammar_mask is None
    assert output.auxiliary_device_output is None
    runner._preprocess_logits.assert_not_called()
    runner.sampler.assert_not_called()
