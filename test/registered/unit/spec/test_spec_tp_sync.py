from __future__ import annotations

import ast
import inspect
import textwrap

import pytest
import torch

from sglang.srt.speculative.spec_tp_sync import SpecTpSync, SpecTpSyncSite


class _FakeTpGroup:
    def __init__(self, *, world_size: int, rank: int, source: torch.Tensor):
        self.world_size = world_size
        self.rank_in_group = rank
        self.source = source
        self.broadcast_calls = 0

    def broadcast(self, values: torch.Tensor, src: int) -> None:
        assert src == 0
        self.broadcast_calls += 1
        values.copy_(self.source)


def _make_sync(group: _FakeTpGroup, sites) -> SpecTpSync:
    sync = SpecTpSync.__new__(SpecTpSync)
    sync._tp_group = group
    sync._sites = frozenset(sites)
    return sync


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_dspark_target_sync_overwrites_rank_local_prefill_sample(device: str):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA-alike GPU is unavailable")

    authoritative = torch.tensor([17, 23, 42], device=device)
    divergent_rank = torch.tensor([17, 99, 42], device=device)
    group = _FakeTpGroup(world_size=8, rank=5, source=authoritative)
    sync = _make_sync(group, {SpecTpSyncSite.DSPARK_TARGET})

    result = sync.sync(SpecTpSyncSite.DSPARK_TARGET, divergent_rank)

    assert result is divergent_rank
    torch.testing.assert_close(result.cpu(), authoritative.cpu())
    assert group.broadcast_calls == 1


@pytest.mark.parametrize(
    ("world_size", "sites"),
    [
        (1, {SpecTpSyncSite.DSPARK_TARGET}),
        (8, set()),
    ],
)
def test_dspark_target_sync_boundary_is_a_noop(world_size, sites):
    values = torch.tensor([3, 5])
    group = _FakeTpGroup(world_size=world_size, rank=0, source=torch.tensor([8, 8]))
    sync = _make_sync(group, sites if world_size > 1 else set())

    result = sync.sync(SpecTpSyncSite.DSPARK_TARGET, values)

    assert result is values
    torch.testing.assert_close(result, torch.tensor([3, 5]))
    assert group.broadcast_calls == 0


def test_dspark_prefill_syncs_sample_before_publishing_sequence_lengths():
    """The v0.5.18 path published a rank-local sample and could drift TP state."""
    from sglang.srt.speculative.dspark_components.dspark_worker_v2 import (
        DSparkWorkerV2,
    )

    tree = ast.parse(textwrap.dedent(inspect.getsource(DSparkWorkerV2._forward_prefill)))
    statements = tree.body[0].body
    sync_index = next(
        i
        for i, node in enumerate(statements)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and ast.unparse(node.value.func) == "self._tp_sync.sync"
        and "SpecTpSyncSite.DSPARK_TARGET" in ast.unparse(node)
        and "next_token_ids" in ast.unparse(node)
    )
    publish_index = next(
        i
        for i, node in enumerate(statements)
        if isinstance(node, ast.Assign)
        and any(ast.unparse(target) == "batch_output.new_seq_lens" for target in node.targets)
    )

    assert sync_index < publish_index
