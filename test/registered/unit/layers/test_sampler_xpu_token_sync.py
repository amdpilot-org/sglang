from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.layers import sampler as sampler_module
from sglang.test.ci.ci_register import register_cpu_ci, register_xpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")
register_xpu_ci(est_time=15, suite="stage-a-test-1-gpu-xpu")


@pytest.mark.parametrize(
    "rank_values, expected",
    [
        ([42, 42], 42),
        ([17, 5], 5),
        ([0, 0, 0, 0], 0),
    ],
)
def test_xpu_token_sync_avoids_integer_min_all_reduce(rank_values, expected):
    token_ids = torch.tensor([rank_values[0]], dtype=torch.long)
    sampler = sampler_module.Sampler.__new__(sampler_module.Sampler)
    sampler.tp_sync_group = object()

    def broken_xccl_all_reduce(tensor, op, group):
        assert op == torch.distributed.ReduceOp.MIN
        tensor.fill_(sum(rank_values))

    def xccl_all_gather(output, tensor, group):
        output.copy_(torch.tensor(rank_values, dtype=tensor.dtype).reshape_as(output))

    with (
        patch.object(sampler_module, "SYNC_TOKEN_IDS_ACROSS_TP", True),
        patch.object(sampler_module, "is_xpu", return_value=True, create=True),
        patch.object(
            torch.distributed, "get_world_size", return_value=len(rank_values)
        ),
        patch.object(
            torch.distributed,
            "all_reduce",
            side_effect=broken_xccl_all_reduce,
        ) as all_reduce,
        patch.object(
            torch.distributed,
            "all_gather_into_tensor",
            side_effect=xccl_all_gather,
        ) as all_gather,
    ):
        sampler._sync_token_ids_across_tp(token_ids, SimpleNamespace(grammars=None))

    assert token_ids.item() == expected
    all_reduce.assert_not_called()
    all_gather.assert_called_once()


def test_non_xpu_token_sync_keeps_min_all_reduce():
    token_ids = torch.tensor([42], dtype=torch.long)
    sampler = sampler_module.Sampler.__new__(sampler_module.Sampler)
    sampler.tp_sync_group = object()

    with (
        patch.object(sampler_module, "SYNC_TOKEN_IDS_ACROSS_TP", True),
        patch.object(sampler_module, "is_xpu", return_value=False, create=True),
        patch.object(torch.distributed, "all_reduce") as all_reduce,
        patch.object(torch.distributed, "all_gather_into_tensor") as all_gather,
    ):
        sampler._sync_token_ids_across_tp(token_ids, SimpleNamespace(grammars=None))

    all_reduce.assert_called_once_with(
        token_ids,
        op=torch.distributed.ReduceOp.MIN,
        group=sampler.tp_sync_group,
    )
    all_gather.assert_not_called()
