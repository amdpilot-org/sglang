from types import SimpleNamespace
from unittest.mock import patch

import pytest

from sglang.srt.distributed import communication_op
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.mark.parametrize(
    ("attn_tp_size", "attn_cp_size", "attn_tp_rank", "attn_cp_rank"),
    [
        (2, 2, 0, 1),  # Regression: a TP source that is not the combined leader.
        (2, 2, 1, 1),
        (2, 1, 1, 0),  # TP-only boundary.
        (1, 2, 0, 1),  # CP-only boundary.
    ],
)
def test_attn_cp_tp_broadcast_stages_data_at_every_source(
    attn_tp_size, attn_cp_size, attn_tp_rank, attn_cp_rank
):
    payload = ["request"]
    initial_data = payload if (attn_tp_rank, attn_cp_rank) == (0, 0) else None
    calls = []

    cp_group = SimpleNamespace(
        name="cp",
        world_size=attn_cp_size,
        rank=attn_tp_rank * attn_cp_size + attn_cp_rank,
        ranks=[attn_tp_rank * attn_cp_size + rank for rank in range(attn_cp_size)],
        cpu_group=object(),
    )
    tp_group = SimpleNamespace(
        name="tp",
        world_size=attn_tp_size,
        rank=attn_tp_rank * attn_cp_size + attn_cp_rank,
        ranks=[rank * attn_cp_size + attn_cp_rank for rank in range(attn_tp_size)],
        cpu_group=object(),
    )

    def fake_broadcast(data, rank, group, src):
        current_group = cp_group if group is cp_group.cpu_group else tp_group
        calls.append(current_group.name)
        if rank == src:
            # Match broadcast_pyobj's source-side requirement: it calls len(data).
            len(data)
        return payload

    with (
        patch.object(communication_op, "get_attn_cp_group", return_value=cp_group),
        patch.object(communication_op, "get_attn_tp_group", return_value=tp_group),
        patch.object(communication_op, "broadcast_pyobj", side_effect=fake_broadcast),
    ):
        result = communication_op.attn_cp_tp_broadcast_pyobj(initial_data)

    assert result == payload
    assert calls == (["cp"] if attn_cp_size > 1 else []) + (
        ["tp"] if attn_tp_size > 1 else []
    )
