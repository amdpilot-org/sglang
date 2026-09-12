from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.srt.layers.sampler import Sampler
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="stage-a-test-cpu-intel")


@pytest.mark.parametrize("grammars", [[object()], None])
def test_token_sync_skips_singleton_group(grammars):
    sampler = Sampler.__new__(Sampler)
    sampler.tp_sync_group = object()
    sampling_info = SimpleNamespace(grammars=grammars)
    token_ids = torch.tensor([17], dtype=torch.int64)

    env_sync = grammars is None
    with (
        patch("sglang.srt.layers.sampler.SYNC_TOKEN_IDS_ACROSS_TP", env_sync),
        patch("sglang.srt.layers.sampler.dist.get_world_size", return_value=1),
        patch("sglang.srt.layers.sampler.dist.all_reduce") as all_reduce,
    ):
        sampler._sync_token_ids_across_tp(token_ids, sampling_info)

    all_reduce.assert_not_called()
    torch.testing.assert_close(token_ids, torch.tensor([17]))


def test_token_sync_preserves_min_reduce_for_multi_rank_group():
    sampler = Sampler.__new__(Sampler)
    sampler.tp_sync_group = object()
    sampling_info = SimpleNamespace(grammars=[object()])
    token_ids = torch.tensor([17], dtype=torch.int64)

    with (
        patch("sglang.srt.layers.sampler.SYNC_TOKEN_IDS_ACROSS_TP", False),
        patch("sglang.srt.layers.sampler.dist.get_world_size", return_value=2),
        patch("sglang.srt.layers.sampler.dist.all_reduce") as all_reduce,
    ):
        sampler._sync_token_ids_across_tp(token_ids, sampling_info)

    all_reduce.assert_called_once_with(
        token_ids,
        op=torch.distributed.ReduceOp.MIN,
        group=sampler.tp_sync_group,
    )


def test_token_sync_does_not_query_group_without_trigger():
    sampler = Sampler.__new__(Sampler)
    sampler.tp_sync_group = object()
    sampling_info = SimpleNamespace(grammars=None)

    with (
        patch("sglang.srt.layers.sampler.SYNC_TOKEN_IDS_ACROSS_TP", False),
        patch("sglang.srt.layers.sampler.dist.get_world_size") as get_world_size,
        patch("sglang.srt.layers.sampler.dist.all_reduce") as all_reduce,
    ):
        sampler._sync_token_ids_across_tp(Mock(), sampling_info)

    get_world_size.assert_not_called()
    all_reduce.assert_not_called()
