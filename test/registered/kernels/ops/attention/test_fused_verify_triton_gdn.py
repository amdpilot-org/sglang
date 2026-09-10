"""Tests for fused sigmoid gating delta rule MTP kernel (GDN target_verify).

Compares the fused kernel `fused_sigmoid_gating_delta_rule_update` against
the reference two-step implementation:
    1. g, beta = fused_gdn_gating(A_log, a, b, dt_bias)
    2. o = fused_recurrent_gated_delta_rule_update(q, k, v, g, beta, ...)
"""

import sys
import torch.nn.functional as F

import pytest
import torch

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

try:
    from sglang.kernels.ops.attention.fla.fused_gdn_gating import fused_gdn_gating
    from sglang.kernels.ops.attention.fla.fused_recurrent import (
        fused_recurrent_gated_delta_rule_update,
    )
    from sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent import (
        fused_sigmoid_gating_delta_rule_update,
    )

    KERNELS_AVAILABLE = True
except ImportError:
    KERNELS_AVAILABLE = False

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=10, suite="nightly-amd-kernel-1-gpu", nightly=True)


def _make_tensors(N, T, H, HV, K, V, device="cuda", seed=2025):
    """Create input tensors for GDN target_verify."""
    torch.manual_seed(seed)
    A_log = torch.randn(HV, dtype=torch.float32, device=device)
    dt_bias = torch.randn(HV, dtype=torch.bfloat16, device=device)
    a = torch.randn(1, N * T, HV, dtype=torch.bfloat16, device=device)
    b = torch.randn(1, N * T, HV, dtype=torch.bfloat16, device=device)
    q = torch.randn(1, N * T, H, K, dtype=torch.bfloat16, device=device)
    k = torch.randn(1, N * T, H, K, dtype=torch.bfloat16, device=device)
    v = torch.randn(1, N * T, HV, V, dtype=torch.bfloat16, device=device)
    indices = torch.arange(N, dtype=torch.int32, device=device)
    initial_state = torch.randn(N, HV, K, V, dtype=torch.float, device=device)
    cu_seqlens = torch.arange(0, N * T + 1, T, dtype=torch.int32, device=device)
    return A_log, dt_bias, a, b, q, k, v, initial_state, indices, cu_seqlens


def run_reference(
    A_log,
    dt_bias,
    q,
    k,
    v,
    a,
    b,
    initial_state_source,
    initial_state_indices,
    cu_seqlens,
    disable_state_update=True,
    intermediate_states_buffer=None,
    intermediate_state_indices=None,
    cache_steps=None,
    retrieve_parent_token=None,
):
    """Reference: fused_gdn_gating + fused_recurrent_gated_delta_rule_update."""
    # fused_gdn_gating expects 2D [seq_len, HV]
    a_2d = a.view(-1, a.shape[-1])
    b_2d = b.view(-1, b.shape[-1])
    g, beta = fused_gdn_gating(A_log, a_2d, b_2d, dt_bias)
    # fused_recurrent expects 3D [B, T, HV]
    g = g.view(a.shape)
    beta = beta.view(b.shape)

    # fused_recurrent requires intermediate_state_indices when cu_seqlens is used
    if cu_seqlens is not None and intermediate_state_indices is None:
        N = len(cu_seqlens) - 1
        intermediate_state_indices = torch.arange(N, dtype=torch.int32, device=q.device)

    return fused_recurrent_gated_delta_rule_update(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state_source=initial_state_source,
        initial_state_indices=initial_state_indices,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=True,
        disable_state_update=disable_state_update,
        intermediate_states_buffer=intermediate_states_buffer,
        intermediate_state_indices=intermediate_state_indices,
        cache_steps=cache_steps,
        retrieve_parent_token=retrieve_parent_token,
    )


def run_fused_mtp(
    A_log,
    dt_bias,
    q,
    k,
    v,
    a,
    b,
    initial_state_source,
    initial_state_indices,
    cu_seqlens,
    disable_state_update=True,
    intermediate_states_buffer=None,
    intermediate_state_indices=None,
    cache_steps=None,
    retrieve_parent_token=None,
):
    """Fused: fused_sigmoid_gating_delta_rule_update."""
    return fused_sigmoid_gating_delta_rule_update(
        A_log=A_log,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        a=a,
        b=b,
        initial_state_source=initial_state_source,
        initial_state_indices=initial_state_indices,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=True,
        softplus_beta=1.0,
        softplus_threshold=20.0,
        is_kda=False,
        disable_state_update=disable_state_update,
        intermediate_states_buffer=intermediate_states_buffer,
        intermediate_state_indices=intermediate_state_indices,
        cache_steps=cache_steps,
        retrieve_parent_token=retrieve_parent_token,
    )


@pytest.mark.skipif(not KERNELS_AVAILABLE, reason="Kernel not available")
@pytest.mark.parametrize("N", [1, 8, 16])
@pytest.mark.parametrize("T", [1, 4, 8])
def test_fused_gdn_mtp_precision(N: int, T: int):
    """Compare fused MTP output against reference."""
    H, HV, K, V = 16, 32, 128, 128

    A_log, dt_bias, a, b, q, k, v, state, indices, cu_seqlens = _make_tensors(
        N, T, H, HV, K, V
    )

    state_ref = state.clone()
    state_fused = state.clone()

    out_ref = run_reference(
        A_log,
        dt_bias,
        q,
        k,
        v,
        a,
        b,
        state_ref,
        indices,
        cu_seqlens,
        disable_state_update=True,
    )
    out_fused = run_fused_mtp(
        A_log,
        dt_bias,
        q,
        k,
        v,
        a,
        b,
        state_fused,
        indices,
        cu_seqlens,
        disable_state_update=True,
    )

    torch.testing.assert_close(out_ref, out_fused, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not KERNELS_AVAILABLE, reason="Kernels not available")
@pytest.mark.parametrize("N", [1, 16, 128])
def test_mtp_single_step_decode(N: int):
    """Verify MTP kernel matches reference for T=1 (decode scenario)."""
    T = 1
    H, HV, K, V = 16, 32, 128, 128

    A_log, dt_bias, a, b, q, k, v, state, indices, cu_seqlens = _make_tensors(
        N, T, H, HV, K, V
    )

    state_ref = state.clone()
    state_fused = state.clone()

    out_ref = run_reference(
        A_log,
        dt_bias,
        q,
        k,
        v,
        a,
        b,
        state_ref,
        indices,
        cu_seqlens,
        disable_state_update=False,
    )
    out_fused = run_fused_mtp(
        A_log,
        dt_bias,
        q,
        k,
        v,
        a,
        b,
        state_fused,
        indices,
        cu_seqlens,
        disable_state_update=False,
    )

    torch.testing.assert_close(out_ref, out_fused, rtol=1e-2, atol=1e-2)

    # Also verify states match after update
    state_diff = (state_ref.float() - state_fused.float()).abs()
    state_max_diff = state_diff.max().item()
    state_fail_rate = (state_diff > 0.1).float().mean().item() * 100
    print(
        f"  single_step state N={N}: max_diff={state_max_diff:.2e}, "
        f"fail_rate={state_fail_rate:.2f}%"
    )
    assert state_fail_rate < 0.01, f"State mismatch: fail_rate={state_fail_rate:.2f}%"


@pytest.mark.skipif(not KERNELS_AVAILABLE, reason="Kernels not available")
def test_verify_scratch_pitch_uses_allocated_steps():
    # Gear below the allocated step dim must not spill into the neighbor block.
    N, T, ALLOCATED = 2, 4, 8
    H, HV, K, V = 16, 32, 128, 128

    A_log, dt_bias, a, b, q, k, v, state, indices, cu_seqlens = _make_tensors(
        N, T, H, HV, K, V
    )
    buffer = torch.full(
        (N + 1, ALLOCATED, HV, V, K), float("nan"), dtype=torch.float32, device="cuda"
    )

    run_fused_mtp(
        A_log,
        dt_bias,
        q,
        k,
        v,
        a,
        b,
        state,
        indices,
        cu_seqlens,
        disable_state_update=True,
        intermediate_states_buffer=buffer,
        intermediate_state_indices=indices,
        cache_steps=T,
    )

    assert not torch.isnan(buffer[:N, :T]).any()
    assert torch.isnan(buffer[N:]).all()
    assert torch.isnan(buffer[:N, T:]).all()


@pytest.mark.skipif(not KERNELS_AVAILABLE, reason="Kernels not available")
@pytest.mark.parametrize(
    "state_dtype",
    [torch.float32, torch.bfloat16, torch.float16],
)
def test_verify_state_dtype_matches_independent_reference(state_dtype):
    N, T, ALLOCATED = 2, 4, 8
    H, HV, K, V = 4, 8, 16, 16
    group = HV // H

    torch.manual_seed(34786)
    A_log = torch.randn(HV, dtype=torch.float32)
    dt_bias = torch.randn(HV, dtype=torch.float32)
    a = torch.randn(1, N * T, HV, dtype=torch.float32)
    b = torch.randn(1, N * T, HV, dtype=torch.float32)
    q = torch.randn(1, N * T, H, K, dtype=torch.float32)
    k = torch.randn(1, N * T, H, K, dtype=torch.float32)
    v = torch.randn(1, N * T, HV, V, dtype=torch.float32)
    state = torch.randn(N, HV, V, K, dtype=torch.float32)

    A_log = A_log.to(device="cuda", dtype=state_dtype)
    dt_bias = dt_bias.to(device="cuda", dtype=torch.bfloat16)
    a = a.to(device="cuda", dtype=torch.bfloat16)
    b = b.to(device="cuda", dtype=torch.bfloat16)
    q = q.to(device="cuda", dtype=torch.bfloat16)
    k = k.to(device="cuda", dtype=torch.bfloat16)
    v = v.to(device="cuda", dtype=torch.bfloat16)
    state = state.to(device="cuda", dtype=state_dtype)
    indices = torch.arange(N, dtype=torch.int32, device="cuda")
    cu_seqlens = torch.arange(0, N * T + 1, T, dtype=torch.int32, device="cuda")

    expected = torch.empty(N, T, HV, V, K, dtype=state_dtype, device="cuda")
    for request_index in range(N):
        hidden_state = state[request_index].float().transpose(-2, -1)
        for step in range(T):
            token_index = request_index * T + step
            query = q[0, token_index].float().repeat_interleave(group, dim=0)
            key = k[0, token_index].float().repeat_interleave(group, dim=0)
            value = v[0, token_index].float()
            gate = -A_log.float().exp() * F.softplus(
                a[0, token_index].float() + dt_bias.float()
            )
            beta = torch.sigmoid(b[0, token_index].float())
            query = query / (query.square().sum(dim=1, keepdim=True) + 1e-6).sqrt()
            key = key / (key.square().sum(dim=1, keepdim=True) + 1e-6).sqrt()
            hidden_state = hidden_state * torch.exp(gate)[:, None, None]
            value = value - (hidden_state * key[:, :, None]).sum(dim=1)
            value = value * beta[:, None]
            hidden_state = hidden_state + key[:, :, None] * value[:, None, :]
            expected[request_index, step] = hidden_state.transpose(-2, -1).to(
                state_dtype
            )

    buffer = torch.full(
        (N + 2, ALLOCATED, HV, V, K),
        float("nan"),
        dtype=state_dtype,
        device="cuda",
    )
    write_indices = torch.tensor([1, 2], dtype=torch.int32, device="cuda")
    buffer_address = buffer.data_ptr()
    assert buffer.stride(0) == ALLOCATED * HV * V * K

    fused_sigmoid_gating_delta_rule_update(
        A_log=A_log,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        a=a,
        b=b,
        initial_state_source=state,
        initial_state_indices=indices,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=True,
        softplus_beta=1.0,
        softplus_threshold=20.0,
        is_kda=False,
        disable_state_update=True,
        intermediate_states_buffer=buffer,
        intermediate_state_indices=write_indices,
        cache_steps=T,
    )
    torch.cuda.synchronize()

    tolerance = {
        torch.float32: (1e-5, 1e-6),
        torch.bfloat16: (2e-2, 2e-3),
        torch.float16: (2e-2, 2e-3),
    }[state_dtype]
    torch.testing.assert_close(
        buffer[1 : N + 1, :T], expected, rtol=tolerance[0], atol=tolerance[1]
    )
    assert torch.isnan(buffer[0]).all()
    assert torch.isnan(buffer[N + 1]).all()
    assert torch.isnan(buffer[1 : N + 1, T:]).all()
    assert buffer.data_ptr() == buffer_address


@pytest.mark.skipif(not KERNELS_AVAILABLE, reason="Kernels not available")
def test_verify_rejects_undocumented_state_dtype():
    N, T, H, HV, K, V = 1, 1, 1, 1, 2, 2
    A_log = torch.zeros(HV, dtype=torch.float32, device="cuda")
    dt_bias = torch.zeros(HV, dtype=torch.bfloat16, device="cuda")
    a = torch.zeros(1, N * T, HV, dtype=torch.bfloat16, device="cuda")
    b = torch.zeros(1, N * T, HV, dtype=torch.bfloat16, device="cuda")
    q = torch.zeros(1, N * T, H, K, dtype=torch.bfloat16, device="cuda")
    k = torch.zeros(1, N * T, H, K, dtype=torch.bfloat16, device="cuda")
    v = torch.zeros(1, N * T, HV, V, dtype=torch.bfloat16, device="cuda")
    state = torch.zeros(N, HV, V, K, dtype=torch.float64, device="cuda")
    buffer = torch.zeros(N, T, HV, V, K, dtype=torch.float64, device="cuda")
    indices = torch.zeros(N, dtype=torch.int32, device="cuda")
    cu_seqlens = torch.tensor([0, 1], dtype=torch.int32, device="cuda")

    with pytest.raises(ValueError, match="Unsupported initial state dtype"):
        fused_sigmoid_gating_delta_rule_update(
            A_log=A_log,
            dt_bias=dt_bias,
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            initial_state_source=state,
            initial_state_indices=indices,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=True,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            is_kda=False,
            disable_state_update=True,
            intermediate_states_buffer=buffer,
            intermediate_state_indices=indices,
            cache_steps=T,
        )


@pytest.mark.skipif(not KERNELS_AVAILABLE, reason="Kernels not available")
def test_verify_rejects_mismatched_state_dtypes():
    N, T, H, HV, K, V = 1, 1, 1, 1, 2, 2
    A_log = torch.zeros(HV, dtype=torch.float32, device="cuda")
    dt_bias = torch.zeros(HV, dtype=torch.bfloat16, device="cuda")
    a = torch.zeros(1, N * T, HV, dtype=torch.bfloat16, device="cuda")
    b = torch.zeros(1, N * T, HV, dtype=torch.bfloat16, device="cuda")
    q = torch.zeros(1, N * T, H, K, dtype=torch.bfloat16, device="cuda")
    k = torch.zeros(1, N * T, H, K, dtype=torch.bfloat16, device="cuda")
    v = torch.zeros(1, N * T, HV, V, dtype=torch.bfloat16, device="cuda")
    state = torch.zeros(N, HV, V, K, dtype=torch.float32, device="cuda")
    buffer = torch.zeros(N, T, HV, V, K, dtype=torch.bfloat16, device="cuda")
    indices = torch.zeros(N, dtype=torch.int32, device="cuda")
    cu_seqlens = torch.tensor([0, 1], dtype=torch.int32, device="cuda")

    with pytest.raises(ValueError, match="state dtypes must match"):
        fused_sigmoid_gating_delta_rule_update(
            A_log=A_log,
            dt_bias=dt_bias,
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            initial_state_source=state,
            initial_state_indices=indices,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=True,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            is_kda=False,
            disable_state_update=True,
            intermediate_states_buffer=buffer,
            intermediate_state_indices=indices,
            cache_steps=T,
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
