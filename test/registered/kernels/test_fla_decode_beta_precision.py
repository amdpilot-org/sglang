"""Regression tests for fp32 beta gates in the FLA decode kernels."""

import pytest
import torch

from sglang.kernels.ops.attention.fla.fused_recurrent import (
    fused_recurrent_gated_delta_rule_packed_decode,
)
from sglang.kernels.ops.attention.fla.fused_recurrent_linear_replayssm import (
    fused_recurrent_linear_replayssm_decode,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=30, stage="stage-b", runner_config="1-gpu-large-amd")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="FLA decode kernels require a GPU"
)


def _inputs(step: int, k_dim: int, v_dim: int):
    # Values are constructed on CPU so the independent reference does not share
    # any Triton implementation details.  Conversion through bf16 models the
    # actual values received by the kernels.
    phase = step + 1
    q = torch.tensor(
        [((i * 7 + phase * 3) % 17 - 8) / 32 for i in range(k_dim)],
        dtype=torch.bfloat16,
    ).float()
    k = torch.tensor(
        [((i * 5 + phase * 2) % 19 - 9) / 32 for i in range(k_dim)],
        dtype=torch.bfloat16,
    ).float()
    v = torch.tensor(
        [((i * 3 + phase * 5) % 23 - 11) / 16 for i in range(v_dim)],
        dtype=torch.bfloat16,
    ).float()
    return q, k, v


@pytest.mark.parametrize("steps", [1, 32, 256])
def test_gdn_packed_decode_keeps_beta_in_fp32(steps):
    k_dim = v_dim = 16
    state = torch.zeros((1, 1, v_dim, k_dim), device="cuda", dtype=torch.float32)
    reference = torch.zeros((v_dim, k_dim), dtype=torch.float32)
    out = torch.empty((1, 1, 1, v_dim), device="cuda", dtype=torch.bfloat16)
    indices = torch.zeros(1, device="cuda", dtype=torch.int32)
    a = torch.full((1, 1), -20.0, device="cuda", dtype=torch.bfloat16)
    b = torch.full((1, 1), 0.5, device="cuda", dtype=torch.bfloat16)
    params = torch.zeros(1, device="cuda", dtype=torch.bfloat16)
    beta = torch.sigmoid(torch.tensor(0.5, dtype=torch.float32))

    for step in range(steps):
        q, k, v = _inputs(step, k_dim, v_dim)
        mixed = torch.cat((q, k, v)).to(device="cuda", dtype=torch.bfloat16)[None]
        fused_recurrent_gated_delta_rule_packed_decode(
            mixed, a, b, params, params, 1.0, state, out, indices
        )
        # a=-20 makes the fp32 decay indistinguishable from one.  Keep every
        # other operation fp32 and only cast the observable output to bf16.
        delta = beta * (v - reference @ k)
        reference = reference + delta[:, None] * k[None, :]
        expected_out = (reference @ q).to(torch.bfloat16)

    torch.testing.assert_close(state[0, 0].cpu(), reference, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(out[0, 0, 0].cpu(), expected_out, rtol=0, atol=0)


@pytest.mark.parametrize("steps", [1, 32, 256])
def test_kda_replay_decode_keeps_beta_in_fp32(steps):
    k_dim = v_dim = 16
    state = torch.zeros((1, 1, v_dim, k_dim), device="cuda", dtype=torch.float32)
    reference = torch.zeros((v_dim, k_dim), dtype=torch.float32)
    out = torch.empty((1, 1, 1, v_dim), device="cuda", dtype=torch.bfloat16)
    d_cache = torch.zeros((1, 1, 1, v_dim), device="cuda", dtype=torch.bfloat16)
    k_cache = torch.zeros((1, 1, 1, k_dim), device="cuda", dtype=torch.bfloat16)
    g_cache = torch.zeros((1, 1, 1, k_dim), device="cuda", dtype=torch.float32)
    indices = torch.zeros(1, device="cuda", dtype=torch.int32)
    write_pos = torch.zeros(1, device="cuda", dtype=torch.int32)
    force_flush = torch.ones(1, device="cuda", dtype=torch.int32)
    a = torch.full((1, 1, k_dim), -20.0, device="cuda", dtype=torch.bfloat16)
    b = torch.full((1, 1), 0.5, device="cuda", dtype=torch.bfloat16)
    A_log = torch.zeros(1, device="cuda", dtype=torch.bfloat16)
    dt_bias = torch.zeros((1, k_dim), device="cuda", dtype=torch.bfloat16)
    beta = torch.sigmoid(torch.tensor(0.5, dtype=torch.float32))

    for step in range(steps):
        q, k, v = _inputs(step, k_dim, v_dim)
        mixed = torch.cat((q, k, v)).to(device="cuda", dtype=torch.bfloat16)[None]
        fused_recurrent_linear_replayssm_decode(
            mixed,
            a,
            b,
            A_log,
            dt_bias,
            1.0,
            state,
            d_cache,
            k_cache,
            g_cache,
            out,
            indices,
            write_pos,
            force_flush=force_flush,
            is_kda=True,
            nk=1,
        )
        delta = beta * (v - reference @ k)
        reference = reference + delta[:, None] * k[None, :]
        expected_out = (reference @ q).to(torch.bfloat16)

    torch.testing.assert_close(state[0, 0].cpu(), reference, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(out[0, 0, 0].cpu(), expected_out, rtol=0, atol=0)
