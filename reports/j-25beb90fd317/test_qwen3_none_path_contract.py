"""Focused reproduction for upstream issue #36820.

These tests execute Qwen3MoeSparseMoeBlock.forward_normal from the prepared
checkout with tiny stand-ins for the gate and experts.  They deliberately do
not claim to implement sparse dispatch; they pin the current a2a=none contract
that every EP rank enters the post-expert reduction, including a rank whose
local expert contribution is identically zero.
"""

from unittest.mock import patch

import pytest
import torch

from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.models.qwen3_moe import Qwen3MoeSparseMoeBlock


class _Gate:
    def __call__(self, hidden_states):
        return torch.zeros((hidden_states.shape[0], 4)), None


class _TopK:
    def __call__(self, hidden_states, router_logits):
        n = hidden_states.shape[0]
        return StandardTopKOutput(
            topk_weights=torch.ones((n, 1)),
            topk_ids=torch.zeros((n, 1), dtype=torch.int64),
            router_logits=router_logits,
        )

    def empty_topk_output(self, device):
        return StandardTopKOutput(
            topk_weights=torch.empty((0, 1), device=device),
            topk_ids=torch.empty((0, 1), dtype=torch.int64, device=device),
            router_logits=torch.empty((0, 4), device=device),
        )


class _Block:
    gate = _Gate()
    topk = _TopK()
    tp_size = 1

    def __init__(self, ep_size, local_value):
        self.ep_size = ep_size
        self.experts = lambda hidden_states, topk_output: torch.full_like(
            hidden_states, local_value
        )


@pytest.mark.parametrize("local_value", [0.0, 3.0])
def test_ep_rank_always_enters_full_reduction(local_value):
    """Zero-contribution and active ranks take the identical collective path."""
    calls = []

    def record_all_reduce(value):
        calls.append(value.clone())
        return value + 10

    block = _Block(ep_size=4, local_value=local_value)
    with (
        patch(
            "sglang.srt.models.qwen3_moe.should_skip_post_experts_all_reduce",
            return_value=False,
        ),
        patch(
            "sglang.srt.models.qwen3_moe.moe_expert_parallel_all_reduce",
            side_effect=record_all_reduce,
        ),
    ):
        result = Qwen3MoeSparseMoeBlock.forward_normal(block, torch.ones((1, 8)))

    assert len(calls) == 1
    assert torch.count_nonzero(calls[0]).item() == (0 if local_value == 0 else 8)
    torch.testing.assert_close(result, torch.full((1, 8), local_value + 10))


def test_ep_size_one_does_not_reduce():
    block = _Block(ep_size=1, local_value=0.0)
    with (
        patch(
            "sglang.srt.models.qwen3_moe.should_skip_post_experts_all_reduce",
            return_value=False,
        ),
        patch(
            "sglang.srt.models.qwen3_moe.moe_expert_parallel_all_reduce"
        ) as all_reduce,
    ):
        result = Qwen3MoeSparseMoeBlock.forward_normal(block, torch.ones((1, 8)))

    all_reduce.assert_not_called()
    torch.testing.assert_close(result, torch.zeros((1, 8)))


def test_existing_fused_skip_avoids_reduction_but_is_not_sparse_dispatch():
    """Guard against mistaking the downstream-fusion skip for issue #36820."""
    block = _Block(ep_size=4, local_value=0.0)
    with (
        patch(
            "sglang.srt.models.qwen3_moe.should_skip_post_experts_all_reduce",
            return_value=True,
        ),
        patch(
            "sglang.srt.models.qwen3_moe.moe_expert_parallel_all_reduce"
        ) as all_reduce,
    ):
        result = Qwen3MoeSparseMoeBlock.forward_normal(block, torch.ones((1, 8)))

    all_reduce.assert_not_called()
    torch.testing.assert_close(result, torch.zeros((1, 8)))
