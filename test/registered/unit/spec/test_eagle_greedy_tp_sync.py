from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.speculative import eagle_utils


class _ReplayBroadcastGroup:
    """Replay rank 0's tensors into later logical ranks."""

    def __init__(self, world_size):
        self.world_size = world_size
        self.rank = 0
        self._rank_zero_values = []
        self.calls = []

    def broadcast(self, tensor, src):
        assert src == 0
        call_index = len(self.calls) % 3
        self.calls.append((self.rank, call_index))
        if self.rank == 0:
            if len(self._rank_zero_values) <= call_index:
                self._rank_zero_values.append(tensor.clone())
            else:
                self._rank_zero_values[call_index] = tensor.clone()
        else:
            tensor.copy_(self._rank_zero_values[call_index])


def _fake_verify_tree_greedy(**kwargs):
    target_predict = kwargs["target_predict"].flatten().to(torch.int32)
    candidates = kwargs["candidates"].flatten().to(torch.int32)
    matches = target_predict.eq(candidates)
    accepted = int(matches.cumprod(dim=0).sum().item())

    predicts = kwargs["predicts"]
    accept_index = kwargs["accept_index"]
    accept_token_num = kwargs["accept_token_num"]
    predicts.copy_(target_predict)
    accept_index.fill_(-1)
    if accepted:
        accept_index[0, :accepted] = torch.arange(
            accepted, dtype=accept_index.dtype, device=accept_index.device
        )
    accept_token_num.fill_(accepted)
    return predicts, accept_index, accept_token_num


def _sample(logits, tp_group, *, dp_attention=False, attn_tp_group=None):
    device = logits.device
    verify_input = SimpleNamespace(
        draft_token_num=2,
        draft_token=torch.tensor([1, 2], dtype=torch.int32, device=device),
        max_tree_depth=2,
        tree_topk=1,
        retrieve_index=torch.zeros((1, 2), dtype=torch.int32, device=device),
        retrieve_next_token=torch.zeros((1, 2), dtype=torch.int32, device=device),
        retrieve_next_sibling=torch.zeros((1, 2), dtype=torch.int32, device=device),
    )
    batch = SimpleNamespace(
        device=device,
        seq_lens=torch.tensor([4], dtype=torch.int32, device=device),
        sampling_info=SimpleNamespace(
            acc_additive_penalties=None,
            acc_scaling_penalties=None,
            logit_bias=None,
            is_all_greedy=True,
        ),
        forward_mode=SimpleNamespace(is_idle=lambda: False),
    )

    with (
        patch.object(eagle_utils, "_is_hip", True),
        patch.object(
            eagle_utils,
            "verify_tree_greedy_func",
            side_effect=_fake_verify_tree_greedy,
        ),
        patch(
            "sglang.srt.layers.dp_attention.is_dp_attention_enabled",
            return_value=dp_attention,
        ),
        patch("sglang.srt.distributed.get_tp_group", return_value=tp_group),
        patch.object(
            eagle_utils,
            "get_parallel",
            return_value=SimpleNamespace(attn_tp_group=attn_tp_group),
        ),
    ):
        return eagle_utils.eagle_sample(
            verify_input,
            batch,
            SimpleNamespace(next_token_logits=logits.clone()),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_rocm_greedy_verify_broadcasts_rank_zero_decision():
    group = _ReplayBroadcastGroup(world_size=2)
    rank_zero_logits = torch.tensor(
        [[0.0, 1.0, 0.0], [0.0, 1.0, 1.0001]], device="cuda"
    )
    rank_one_logits = torch.tensor(
        [[0.0, 1.0, 0.0], [0.0, 1.0001, 1.0]], device="cuda"
    )

    group.rank = 0
    rank_zero = _sample(rank_zero_logits, group)
    group.rank = 1
    rank_one = _sample(rank_one_logits, group)

    assert torch.argmax(rank_zero_logits, dim=-1).tolist() == [1, 2]
    assert torch.argmax(rank_one_logits, dim=-1).tolist() == [1, 1]
    assert len(group.calls) == 6
    for rank_zero_tensor, rank_one_tensor in zip(rank_zero, rank_one):
        torch.testing.assert_close(rank_one_tensor, rank_zero_tensor)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_rocm_greedy_verify_tp_one_keeps_local_decision():
    group = _ReplayBroadcastGroup(world_size=1)
    logits = torch.tensor(
        [[0.0, 1.0, 0.0], [0.0, 1.0001, 1.0]], device="cuda"
    )

    predict, num_correct_drafts, _ = _sample(logits, group)

    assert group.calls == []
    assert predict.tolist() == [1, 1]
    # eagle_sample returns accepted drafts plus the trailing bonus token.
    assert num_correct_drafts.tolist() == [2]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_rocm_greedy_verify_uses_attention_tp_group_with_dp_attention():
    regular_group = _ReplayBroadcastGroup(world_size=1)
    attention_group = _ReplayBroadcastGroup(world_size=2)
    logits = torch.tensor(
        [[0.0, 1.0, 0.0], [0.0, 1.0, 1.0001]], device="cuda"
    )

    _sample(
        logits,
        regular_group,
        dp_attention=True,
        attn_tp_group=attention_group,
    )

    assert regular_group.calls == []
    assert attention_group.calls == [(0, 0), (0, 1), (0, 2)]
