import pytest
import torch
import triton

from sglang.kernels.ops.moe.ep_moe_kernels import _fwd_kernel_ep_scatter_1


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA or ROCm is required"
)


CASES = (
    ([128], [0]),
    ([128, 128, 128], [0, 128, 127]),
    ([128, 256, 0, 128], [100, 200, 0, 128]),
    ([0, 128, 128, 0, 128], [0, 128, 128, 0, 127]),
)


def _reference(padded_counts, valid_counts):
    padded_counts = torch.tensor(padded_counts, dtype=torch.int32)
    valid_counts = torch.tensor(valid_counts, dtype=torch.int32)
    expected_starts = torch.cumsum(padded_counts, dim=0) - padded_counts
    expected_m_indices = torch.full(
        (int(padded_counts.sum().item()),), -1, dtype=torch.int32
    )

    for expert, valid_count in enumerate(valid_counts.tolist()):
        if valid_count:
            indices = expected_starts[expert] + torch.arange(valid_count)
            expected_m_indices.scatter_(0, indices.long(), expert)

    return expected_starts, expected_m_indices


@pytest.mark.parametrize(("padded_counts", "valid_counts"), CASES)
def test_fwd_kernel_ep_scatter_1_matches_torch_scatter_and_preserves_guards(
    padded_counts, valid_counts
):
    device = torch.device("cuda")
    num_experts = len(padded_counts)
    total_tokens = sum(padded_counts)
    guard_size = max(padded_counts)

    assert all(count % 128 == 0 for count in padded_counts)
    assert all(valid <= padded for valid, padded in zip(valid_counts, padded_counts))

    counts = torch.tensor(padded_counts, dtype=torch.int32, device=device)
    valid = torch.tensor(valid_counts, dtype=torch.int32, device=device)
    starts = torch.full((num_experts,), -1, dtype=torch.int32, device=device)
    storage = torch.full(
        (guard_size + total_tokens + guard_size,), -7, dtype=torch.int32, device=device
    )
    m_indices = storage[guard_size : guard_size + total_tokens]

    _fwd_kernel_ep_scatter_1[(num_experts,)](
        counts,
        valid,
        starts,
        m_indices,
        num_experts=num_experts,
        num_warps=8,
        BLOCK_E=128,
        BLOCK_EXPERT_NUM=triton.next_power_of_2(num_experts),
    )
    torch.cuda.synchronize()

    expected_starts, expected_m_indices = _reference(padded_counts, valid_counts)
    assert torch.equal(starts.cpu(), expected_starts)
    assert torch.equal(m_indices.cpu(), expected_m_indices)
    assert torch.all(storage[:guard_size].cpu() == -7)
    assert torch.all(storage[guard_size + total_tokens :].cpu() == -7)
