import pytest
import torch
import torch.nn.functional as F

from sglang.kernels.ops.mamba.selective_scan import selective_scan_fn
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd")


def reference_scan(u, delta, A, B, C, D, z, delta_bias, initial_state):
    state = initial_state.float().clone()
    ys = []
    for token in range(u.shape[1]):
        dt = F.softplus(delta[:, token].float() + delta_bias.float())
        state = (
            torch.exp(dt[..., None] * A.float()[None]) * state
            + dt[..., None]
            * B[:, token].float()[:, None]
            * u[:, token].float()[..., None]
        )
        y = torch.einsum("bdn,bn->bd", state, C[:, token].float())
        ys.append((y + D.float() * u[:, token].float()) * F.silu(z[:, token].float()))
    return torch.stack(ys, 1).to(u.dtype), state


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_mamba1_selective_scan_matches_independent_gpu_reference(dtype):
    if not torch.cuda.is_available():
        pytest.skip("requires one GPU")
    torch.manual_seed(17)
    device = "cuda"
    batch, length, dim, state_size = 2, 7, 12, 5
    u = torch.randn(batch, length, dim, device=device, dtype=dtype)
    delta = torch.randn_like(u)
    A = -torch.rand(dim, state_size, device=device)
    B = torch.randn(batch, length, state_size, device=device, dtype=dtype)
    C = torch.randn_like(B)
    D = torch.randn(dim, device=device)
    z = torch.randn_like(u)
    bias = torch.randn(dim, device=device)
    initial = torch.randn(batch, dim, state_size, device=device)

    expected, expected_state = reference_scan(u, delta, A, B, C, D, z, bias, initial)
    actual, actual_state = selective_scan_fn(
        u,
        delta,
        A,
        B,
        C,
        D=D,
        z=z,
        delta_bias=bias,
        delta_softplus=True,
        return_last_state=True,
        initial_state=initial,
    )
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(actual_state, expected_state, rtol=1e-5, atol=1e-5)


def test_mamba1_selective_scan_rejects_unpacked_input():
    with pytest.raises(ValueError, match="batch, length, dim"):
        selective_scan_fn(*(torch.zeros(2, 3) for _ in range(5)))
