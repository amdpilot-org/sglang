from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.mark.parametrize(
    "values, expected",
    [
        ([[]], [[]]),
        ([torch.tensor([0.25, -1.5])], [[0.25, -1.5]]),
        (
            [[], torch.tensor([2.0, 3.0])],
            [[], [2.0, 3.0]],
        ),
    ],
)
def test_move_logprobs_to_cpu_accepts_host_lists_and_tensors(values, expected):
    processor = object.__new__(SchedulerBatchResultProcessor)
    output = LogitsProcessorOutput(
        next_token_logits=None,
        next_token_token_ids_logprobs_val=values,
    )

    processor.move_logprobs_to_cpu(
        batch=SimpleNamespace(return_logprob=True), logits_output=output
    )

    assert output.next_token_token_ids_logprobs_val == expected


def test_normalize_decode_outputs_accepts_mixed_token_id_logprob_values():
    processor = object.__new__(SchedulerBatchResultProcessor)
    output = LogitsProcessorOutput(
        next_token_logits=None,
        next_token_logprobs=torch.tensor([-0.5]),
        next_token_token_ids_logprobs_val=[
            [],
            torch.tensor([2.0, 3.0]),
        ],
    )
    batch = SimpleNamespace(
        return_logprob=True,
        spec_algorithm=SimpleNamespace(is_none=lambda: True),
    )

    next_token_ids, next_token_logprobs = processor._normalize_decode_outputs(
        batch=batch,
        result=SimpleNamespace(),
        logits_output=output,
        next_token_ids=[7],
    )

    assert next_token_ids == [[7]]
    assert next_token_logprobs == [-0.5]
    assert output.next_token_token_ids_logprobs_val == [
        [],
        [2.0, 3.0],
    ]
