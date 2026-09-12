from types import SimpleNamespace

import pytest
import torch

from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


def _logits_output(**overrides):
    values = dict(
        next_token_logprobs=torch.tensor([-0.1, -0.2]),
        input_token_logprobs=None,
        next_token_top_logprobs_val=None,
        next_token_top_logprobs_idx=None,
        next_token_token_ids_logprobs_val=None,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _normalize_decode_outputs(logits_output):
    batch = SimpleNamespace(
        return_logprob=True,
        spec_algorithm=SimpleNamespace(is_none=lambda: True),
    )
    return SchedulerBatchResultProcessor._normalize_decode_outputs(
        None,
        batch=batch,
        result=SimpleNamespace(),
        logits_output=logits_output,
        next_token_ids=torch.tensor([1, 2]),
    )


def test_move_logprobs_to_cpu_accepts_mixed_tensor_and_list_entries():
    logits_output = _logits_output(
        next_token_top_logprobs_val=[torch.tensor([-0.5]), []],
        next_token_top_logprobs_idx=[torch.tensor([7]), []],
        next_token_token_ids_logprobs_val=[torch.tensor([-0.25]), []],
    )

    SchedulerBatchResultProcessor.move_logprobs_to_cpu(
        None,
        batch=SimpleNamespace(return_logprob=True),
        logits_output=logits_output,
    )

    assert logits_output.next_token_top_logprobs_val == [[-0.5], []]
    assert logits_output.next_token_top_logprobs_idx == [[7], []]
    assert logits_output.next_token_token_ids_logprobs_val == [[-0.25], []]


def test_normalize_decode_outputs_accepts_mixed_tensor_and_list_entries():
    logits_output = _logits_output(
        next_token_top_logprobs_val=[torch.tensor([-0.5]), []],
        next_token_top_logprobs_idx=[torch.tensor([7]), []],
        next_token_token_ids_logprobs_val=[torch.tensor([-0.25]), []],
    )

    next_token_ids, next_token_logprobs = _normalize_decode_outputs(logits_output)

    assert next_token_ids == [[1], [2]]
    assert next_token_logprobs == pytest.approx([-0.1, -0.2])
    assert logits_output.next_token_top_logprobs_val == [[-0.5], []]
    assert logits_output.next_token_top_logprobs_idx == [[7], []]
    assert logits_output.next_token_token_ids_logprobs_val == [[-0.25], []]


def test_move_logprobs_to_cpu_preserves_already_host_side_entries():
    logits_output = _logits_output(
        next_token_top_logprobs_val=[[-0.5], []],
        next_token_top_logprobs_idx=[[7], []],
        next_token_token_ids_logprobs_val=[[-0.25], []],
    )

    SchedulerBatchResultProcessor.move_logprobs_to_cpu(
        None,
        batch=SimpleNamespace(return_logprob=True),
        logits_output=logits_output,
    )

    assert logits_output.next_token_top_logprobs_val == [[-0.5], []]
    assert logits_output.next_token_top_logprobs_idx == [[7], []]
    assert logits_output.next_token_token_ids_logprobs_val == [[-0.25], []]


def test_normalize_decode_outputs_still_converts_all_tensor_entries():
    logits_output = _logits_output(
        next_token_top_logprobs_val=[torch.tensor([-0.5]), torch.empty(0)],
        next_token_top_logprobs_idx=[
            torch.tensor([7]),
            torch.empty(0, dtype=torch.long),
        ],
        next_token_token_ids_logprobs_val=[torch.tensor([-0.25]), torch.empty(0)],
    )

    _normalize_decode_outputs(logits_output)

    assert logits_output.next_token_top_logprobs_val == [[-0.5], []]
    assert logits_output.next_token_top_logprobs_idx == [[7], []]
    assert logits_output.next_token_token_ids_logprobs_val == [[-0.25], []]
