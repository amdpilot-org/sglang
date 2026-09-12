from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.sampler import Sampler


def _sampler(entry_only: bool) -> Sampler:
    sampler = Sampler.__new__(Sampler)
    sampler.tp_sync_group = object()
    sampler.tp_grammar_entry_only = entry_only
    return sampler


def test_entry_only_grammar_tokens_are_broadcast_from_group_root():
    sampler = _sampler(entry_only=True)
    token_ids = torch.tensor([7, 11], dtype=torch.int64)

    with patch("torch.distributed.broadcast") as broadcast, patch(
        "torch.distributed.all_reduce"
    ) as all_reduce:
        sampler._sync_token_ids_across_tp(
            token_ids, SimpleNamespace(grammars=[object()])
        )

    broadcast.assert_called_once_with(
        token_ids, group_src=0, group=sampler.tp_sync_group
    )
    all_reduce.assert_not_called()


def test_grammar_fallback_keeps_min_all_reduce():
    sampler = _sampler(entry_only=False)
    token_ids = torch.tensor([7], dtype=torch.int64)

    with patch("torch.distributed.broadcast") as broadcast, patch(
        "torch.distributed.all_reduce"
    ) as all_reduce:
        sampler._sync_token_ids_across_tp(
            token_ids, SimpleNamespace(grammars=[object()])
        )

    broadcast.assert_not_called()
    all_reduce.assert_called_once_with(
        token_ids,
        op=torch.distributed.ReduceOp.MIN,
        group=sampler.tp_sync_group,
    )


def test_unconstrained_batch_adds_no_collective():
    sampler = _sampler(entry_only=True)
    token_ids = torch.tensor([7], dtype=torch.int64)

    with patch("torch.distributed.broadcast") as broadcast, patch(
        "torch.distributed.all_reduce"
    ) as all_reduce:
        sampler._sync_token_ids_across_tp(token_ids, SimpleNamespace(grammars=None))

    broadcast.assert_not_called()
    all_reduce.assert_not_called()
