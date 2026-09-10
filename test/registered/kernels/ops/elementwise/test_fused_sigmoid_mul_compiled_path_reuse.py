import sys
import time

import pytest
import torch

from sglang.kernels.ops.elementwise.elementwise import (
    _fused_sigmoid_mul_kernel,
    fused_sigmoid_mul,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=8, stage="jit-kernel-unit", runner_config="amd")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="fused sigmoid-mul reuse tests require CUDA or ROCm.",
)

DEVICE = "cuda"
DTYPE = torch.float16
SENTINEL = -12345.0
SHAPE_SEQUENCE = ((1, 2048), (3, 2048), (7, 2048))


def _reference(attn_output: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    return (
        attn_output.detach()
        .cpu()
        .double()
        .mul(torch.sigmoid(gate.detach().cpu().double()))
        .to(DTYPE)
    )


def _cache_size() -> int:
    device_cache = _fused_sigmoid_mul_kernel.device_caches.get(0)
    if not device_cache:
        return 0
    return len(device_cache[0])


def _timed_call(attn_output: torch.Tensor, gate: torch.Tensor) -> tuple[torch.Tensor, float]:
    torch.cuda.synchronize()
    started_ns = time.perf_counter_ns()
    output = fused_sigmoid_mul(attn_output, gate, inplace=False)
    torch.cuda.synchronize()
    return output, (time.perf_counter_ns() - started_ns) / 1_000_000


def test_fused_sigmoid_mul_cold_warm_compiled_path_reuse() -> None:
    cache_before = _cache_size()
    timings_ms = {}

    for shape in SHAPE_SEQUENCE:
        generator = torch.Generator(device="cpu").manual_seed(20260910)
        attn_output = torch.randn(
            shape, generator=generator, dtype=torch.float32
        ).to(device=DEVICE, dtype=DTYPE)
        gate = torch.randn(
            shape, generator=generator, dtype=torch.float32
        ).to(device=DEVICE, dtype=DTYPE)
        original_attn = attn_output.detach().cpu().clone()
        original_gate = gate.detach().cpu().clone()
        expected = _reference(attn_output, gate)

        output, elapsed_ms = _timed_call(attn_output, gate)
        timings_ms[shape] = elapsed_ms

        assert output.shape == shape
        assert output.dtype is DTYPE
        assert output.data_ptr() != attn_output.data_ptr()
        torch.testing.assert_close(
            output.cpu(), expected, rtol=1.0e-2, atol=1.0e-2
        )
        assert torch.equal(attn_output.cpu(), original_attn)
        assert torch.equal(gate.cpu(), original_gate)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                fused_sigmoid_mul(attn_output, gate, inplace=False)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output = fused_sigmoid_mul(attn_output, gate, inplace=False)

        static_addresses = (
            attn_output.data_ptr(),
            gate.data_ptr(),
            graph_output.data_ptr(),
        )
        guard = torch.full(shape, SENTINEL, device=DEVICE, dtype=DTYPE)
        graph_output.fill_(SENTINEL)
        torch.cuda.synchronize()
        for _ in range(4):
            graph.replay()
        torch.cuda.synchronize()

        assert static_addresses == (
            attn_output.data_ptr(),
            gate.data_ptr(),
            graph_output.data_ptr(),
        )
        assert graph_output.dtype is DTYPE
        torch.testing.assert_close(
            graph_output.cpu(), expected, rtol=1.0e-2, atol=1.0e-2
        )
        assert guard.cpu().eq(SENTINEL).all()
        assert torch.equal(attn_output.cpu(), original_attn)
        assert torch.equal(gate.cpu(), original_gate)

    cache_after = _cache_size()
    assert cache_after - cache_before == 1
    print(f"fused_sigmoid_mul_reuse_timings_ms={timings_ms}")


def test_fused_sigmoid_mul_preserves_documented_inplace_aliasing() -> None:
    shape = SHAPE_SEQUENCE[0]
    generator = torch.Generator(device="cpu").manual_seed(20260911)
    attn_output = torch.randn(
        shape, generator=generator, dtype=torch.float32
    ).to(device=DEVICE, dtype=DTYPE)
    gate = torch.randn(
        shape, generator=generator, dtype=torch.float32
    ).to(device=DEVICE, dtype=DTYPE)
    original_gate = gate.detach().cpu().clone()
    expected = _reference(attn_output, gate)
    output_address = attn_output.data_ptr()

    output = fused_sigmoid_mul(attn_output, gate, inplace=True)

    assert output.data_ptr() == output_address
    assert output.dtype is DTYPE
    torch.testing.assert_close(
        output.cpu(), expected, rtol=1.0e-2, atol=1.0e-2
    )
    assert torch.equal(gate.cpu(), original_gate)


def test_fused_sigmoid_mul_rejects_mismatched_shape() -> None:
    attn_output = torch.randn((2, 2048), device=DEVICE, dtype=DTYPE)
    gate = torch.randn((3, 2048), device=DEVICE, dtype=DTYPE)

    with pytest.raises(AssertionError, match="attn_output and gate must have the same shape"):
        fused_sigmoid_mul(attn_output, gate, inplace=False)


def test_fused_sigmoid_mul_rejects_complex_dtype() -> None:
    attn_output = torch.randn((1, 2048), device=DEVICE, dtype=DTYPE)
    gate = torch.randn((1, 2048), device=DEVICE, dtype=DTYPE)
    complex_attn = torch.complex(attn_output, attn_output)
    complex_gate = torch.complex(gate, gate)

    with pytest.raises(KeyError, match="complex32"):
        fused_sigmoid_mul(complex_attn, complex_gate, inplace=False)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
