"""Regression tests for DFlash/DSpark verify-time logit adjustments."""

from types import SimpleNamespace

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.speculative.dflash_utils import (
    apply_dflash_verify_logits_adjustments,
)
from sglang.srt.speculative.dspark_components.dspark_verify import (
    verify_logits_adjustments_are_noop,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _SamplingInfo(SimpleNamespace):
    def __len__(self) -> int:
        return self.batch_size


def _sampling_info(*, batch_size: int, additive_penalties=None, logit_bias=None):
    return _SamplingInfo(
        batch_size=batch_size,
        has_custom_logit_processor=False,
        acc_additive_penalties=additive_penalties,
        penalizer_orchestrator=None,
        grammar_mask=None,
        logit_bias=logit_bias,
    )


@pytest.mark.parametrize("draft_token_num", [1, 3])
def test_broadcasts_additive_penalties_over_entire_verify_block(draft_token_num):
    additive_penalties = torch.tensor(
        [[0.0, float("-inf"), 2.0], [3.0, 0.5, -4.0]], dtype=torch.float32
    )
    sampling_info = _sampling_info(batch_size=2, additive_penalties=additive_penalties)
    logits = torch.zeros((2 * draft_token_num, 3), dtype=torch.float64)

    apply_dflash_verify_logits_adjustments(
        next_token_logits=logits,
        sampling_info=sampling_info,
        draft_token_num=draft_token_num,
    )

    expected = (
        additive_penalties.to(dtype=logits.dtype)
        .unsqueeze(1)
        .expand(-1, draft_token_num, -1)
        .reshape_as(logits)
    )
    torch.testing.assert_close(logits, expected)


def test_additive_penalties_compose_with_logit_bias():
    additive_penalties = torch.tensor([[0.0, -2.0, 1.0]])
    logit_bias = torch.tensor([[4.0, 0.5, -3.0]])
    sampling_info = _sampling_info(
        batch_size=1,
        additive_penalties=additive_penalties,
        logit_bias=logit_bias,
    )
    logits = torch.zeros((2, 3))

    apply_dflash_verify_logits_adjustments(
        next_token_logits=logits,
        sampling_info=sampling_info,
        draft_token_num=2,
    )

    torch.testing.assert_close(
        logits, (additive_penalties + logit_bias).expand_as(logits)
    )


def test_additive_penalties_disable_dspark_noop_fast_path():
    sampling_info = _sampling_info(
        batch_size=1,
        additive_penalties=torch.zeros((1, 4), dtype=torch.float32),
    )

    assert not verify_logits_adjustments_are_noop(sampling_info)


def test_absent_adjustments_keep_dspark_noop_fast_path():
    assert verify_logits_adjustments_are_noop(_sampling_info(batch_size=1))
