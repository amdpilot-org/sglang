import ast
import inspect
import textwrap

import pytest
import torch
import triton

from sglang.kernels.ops.moe.ep_moe_kernels import _fwd_kernel_ep_scatter_1
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=8, stage="base-b-kernel-unit", runner_config="1-gpu-l40s")


def test_ep_scatter_prefix_is_not_reloaded():
    """The prefix producer and consumer must not communicate without a barrier."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(_fwd_kernel_ep_scatter_1.fn)))

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "tl"
            and node.func.attr == "load"
        ):
            continue
        assert not any(
            isinstance(child, ast.Name) and child.id == "expert_start_loc"
            for child in ast.walk(node)
        ), "expert_start_loc is written and then reloaded without a CTA barrier"


@pytest.mark.parametrize(
    "padded_counts,valid_counts,block_e",
    [
        ([0], [0], 128),
        ([0, 128, 0, 256, 128], [0, 1, 0, 255, 127], 128),
        ([64, 192, 64], [64, 129, 0], 64),
        ([32] * 17, [0, 1, 31, 32] * 4 + [17], 32),
    ],
)
def test_ep_scatter_prefix_and_indices(padded_counts, valid_counts, block_e):
    padded = torch.tensor(padded_counts, dtype=torch.int32, device="cuda")
    valid = torch.tensor(valid_counts, dtype=torch.int32, device="cuda")
    starts = torch.full_like(padded, -1)
    m_indices = torch.full((sum(padded_counts),), -2, dtype=torch.int32, device="cuda")

    handle = _fwd_kernel_ep_scatter_1[(len(padded_counts),)](
        padded,
        valid,
        starts,
        m_indices,
        num_experts=len(padded_counts),
        BLOCK_E=block_e,
        BLOCK_EXPERT_NUM=triton.next_power_of_2(len(padded_counts)),
        num_warps=8,
    )

    expected_starts = (torch.cumsum(padded, dim=0) - padded).to(torch.int32)
    expected_indices = torch.cat(
        [
            torch.cat(
                [
                    torch.full((valid_count,), expert, dtype=torch.int32),
                    torch.full((padded_count - valid_count,), -1, dtype=torch.int32),
                ]
            )
            for expert, (padded_count, valid_count) in enumerate(
                zip(padded_counts, valid_counts)
            )
        ]
    ).to("cuda")

    torch.testing.assert_close(starts, expected_starts, rtol=0, atol=0)
    torch.testing.assert_close(m_indices, expected_indices, rtol=0, atol=0)
    assert "tt.load %cur_expert_start" not in handle.asm["ttir"]
