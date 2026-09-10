import pytest
import torch

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=2, stage="base-b", runner_config="1-gpu-large")
register_amd_ci(est_time=2, stage="jit-kernel-unit", runner_config="amd")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="in-place cycle permutation tests require CUDA or ROCm.",
)

DEVICE = "cuda"
DTYPE = torch.bfloat16


def _permute_rows_in_place(
    rows: torch.Tensor,
    permutation: list[int],
    scratch: torch.Tensor,
) -> None:
    active_count = len(permutation)
    assert rows.ndim >= 2
    assert rows.shape[0] >= active_count
    assert scratch.shape == rows.shape[1:]
    assert scratch.device == rows.device
    assert sorted(permutation) == list(range(active_count))

    remaining = set(range(active_count))
    while remaining:
        start = min(remaining)
        cycle = [start]
        current = start
        while permutation[current] != start:
            current = permutation[current]
            cycle.append(current)
            remaining.remove(current)
        remaining.remove(start)

        scratch.copy_(rows[cycle[0]])
        for current, next_index in zip(cycle[:-1], cycle[1:]):
            rows[current].copy_(rows[next_index])
        rows[cycle[-1]].copy_(scratch)


@pytest.mark.parametrize(
    "permutation",
    [
        [0, 1, 2, 3],
        [1, 0, 3, 2],
        [1, 2, 0],
        [1, 2, 3, 0],
        [0, 2, 1, 4, 3, 5],
    ],
    ids=[
        "fixed_points",
        "two_cycles",
        "three_cycle",
        "four_cycle",
        "mixed_cycles",
    ],
)
def test_inplace_cycle_permutation(permutation: list[int]) -> None:
    active_count = len(permutation)
    rows = torch.randn(
        (active_count + 2, 2, 8), dtype=DTYPE, device=DEVICE
    )
    original = rows.clone()
    indices = torch.tensor(permutation, dtype=torch.int64, device=DEVICE)
    expected = original[:active_count].index_select(0, indices)
    unused = original[active_count:].clone()
    scratch = torch.empty_like(rows[0])
    permutation_plan = list(permutation)

    scratch.copy_(rows[0])
    torch.cuda.synchronize()
    allocated_before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    _permute_rows_in_place(rows, permutation_plan, scratch)
    torch.cuda.synchronize()

    assert torch.cuda.memory_allocated() == allocated_before
    assert torch.cuda.max_memory_allocated() == allocated_before
    assert torch.equal(rows[:active_count], expected)
    assert torch.equal(rows[active_count:], unused)
