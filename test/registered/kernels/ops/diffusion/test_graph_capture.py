"""Graph-capture semantics for the diffusion RMSNorm + SiLU-mul chain."""

import statistics
import sys
import time

import pytest
import torch
import torch.nn.functional as F

from sglang.kernels.ops.diffusion.activation.silu_mul_bitexact import (
    can_use_fused_silu_mul,
    fused_silu_mul_bitexact,
)
from sglang.kernels.ops.diffusion.norm.native_bf16_rmsnorm_triton import (
    MAX_HIDDEN_SIZE,
    rmsnorm_scale,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=10, stage="jit-kernel-unit", runner_config="amd")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")

DEVICE = "cuda"
EPS = 1e-6
SHAPES = [(1, 4096), (16, 4096), (32, 4096)]


@pytest.fixture(autouse=True)
def cuda_setup():
    torch.cuda.manual_seed(0)


def _native_bf16_rmsnorm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    square = (x * x).to(torch.bfloat16)
    mean_square = square.mean(dim=-1, keepdim=True).to(torch.bfloat16)
    rstd = torch.rsqrt((mean_square + EPS).to(torch.bfloat16).float()).to(
        torch.bfloat16
    )
    return ((x * rstd).to(torch.bfloat16) * weight).to(torch.bfloat16)


def _chain(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    norm = rmsnorm_scale(x, weight, scale, EPS)
    assert norm is not None
    return fused_silu_mul_bitexact(norm, gate)


def _reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    norm = (_native_bf16_rmsnorm(x, weight) * scale).to(torch.bfloat16)
    return F.silu(norm) * gate


def _capture_chain(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    gate: torch.Tensor,
) -> tuple[torch.cuda.CUDAGraph, torch.Tensor]:
    capture_stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        output = _chain(x, weight, scale, gate)
    return graph, output


def _bounded_latency_ms(fn) -> list[float]:
    for _ in range(3):
        fn()
    torch.cuda.current_stream().synchronize()
    samples = []
    for _ in range(20):
        start = time.perf_counter()
        fn()
        torch.cuda.current_stream().synchronize()
        samples.append((time.perf_counter() - start) * 1e3)
    assert max(samples) < 50.0
    return samples


@torch.inference_mode()
@pytest.mark.parametrize("rows,hidden", SHAPES)
def test_rmsnorm_silu_chain_eager_graph_replay_and_bounded_latency(
    rows: int, hidden: int
):
    x = torch.randn(rows, hidden, device=DEVICE, dtype=torch.bfloat16)
    weight = torch.randn(hidden, device=DEVICE, dtype=torch.bfloat16)
    scale = torch.randn(rows, hidden, device=DEVICE, dtype=torch.bfloat16)
    gate = torch.randn_like(scale)

    eager = _chain(x, weight, scale, gate)
    reference = _reference(x, weight, scale, gate)
    torch.testing.assert_close(eager, reference, atol=2e-2, rtol=2e-2)

    graph, captured = _capture_chain(x, weight, scale, gate)
    graph.replay()
    torch.cuda.current_stream().synchronize()
    assert torch.equal(captured, eager)

    expected_x = x.mul(1.25)
    expected_scale = scale.mul(0.75)
    expected_gate = gate.mul(0.5)
    x.mul_(1.25)
    scale.mul_(0.75)
    gate.mul_(0.5)
    graph.replay()
    torch.cuda.current_stream().synchronize()

    assert torch.equal(x, expected_x)
    assert torch.equal(scale, expected_scale)
    assert torch.equal(gate, expected_gate)
    torch.testing.assert_close(
        captured,
        _reference(x, weight, scale, gate),
        atol=2e-2,
        rtol=2e-2,
    )

    eager_samples = _bounded_latency_ms(lambda: _chain(x, weight, scale, gate))
    graph_samples = _bounded_latency_ms(graph.replay)
    assert statistics.median(eager_samples) < 50.0
    assert statistics.median(graph_samples) < 50.0


@torch.inference_mode()
def test_rmsnorm_silu_capture_support_boundaries():
    hidden = MAX_HIDDEN_SIZE + 1
    x = torch.randn(2, hidden, device=DEVICE, dtype=torch.bfloat16)
    weight = torch.randn(hidden, device=DEVICE, dtype=torch.bfloat16)
    scale = torch.randn_like(x)
    gate = torch.randn_like(x)

    assert rmsnorm_scale(x, weight, scale, EPS) is None
    assert rmsnorm_scale(x.cpu(), weight, scale, EPS) is None
    assert not can_use_fused_silu_mul(x, gate.float())
    assert not can_use_fused_silu_mul(x, gate[:, :-1])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
