from __future__ import annotations

import pytest
import torch

from sglang.kernels.fused_op import (
    clear_fused_op_trace,
    disable_fused_op_trace,
    enable_fused_op_trace,
    get_fused_op_backend,
    get_fused_op_trace,
    set_fused_op_backend,
)
from sglang.kernels.spec import KernelBackend
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.model_executor.runner import model_capture_mode
from sglang.srt.models.olmo2 import Olmo2Attention
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd")


def _reference(input: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    input64 = input.double()
    weight64 = weight.double()
    return (
        input64
        * torch.rsqrt(input64.square().mean(dim=-1, keepdim=True) + eps)
        * weight64
    ).to(input.dtype)


def _make_attention(q_norm: RMSNorm, k_norm: RMSNorm) -> Olmo2Attention:
    attention = object.__new__(Olmo2Attention)
    torch.nn.Module.__init__(attention)
    attention.tp_size = 1
    attention.tp_rank = 0
    attention.alt_stream = torch.cuda.Stream()
    attention.q_norm = q_norm
    attention.k_norm = k_norm
    return attention


def _padded_tensor(
    shape: tuple[int, int], dtype: torch.dtype, device: torch.device, sentinel: float
) -> tuple[torch.Tensor, torch.Tensor]:
    numel = shape[0] * shape[1]
    flat = torch.full((numel + 2,), sentinel, dtype=dtype, device=device)
    return flat, flat[1 : 1 + numel].view(shape)


def _assert_sentinels(
    flat_q: torch.Tensor,
    flat_k: torch.Tensor,
    guard_before: torch.Tensor,
    guard_after: torch.Tensor,
    sentinel: float,
) -> None:
    expected_q_before = torch.full_like(flat_q[:1], sentinel)
    expected_q_after = torch.full_like(flat_q[-1:], sentinel)
    expected_k_before = torch.full_like(flat_k[:1], sentinel)
    expected_k_after = torch.full_like(flat_k[-1:], sentinel)
    expected_guard = torch.full_like(guard_before, sentinel)
    assert torch.equal(flat_q[:1], expected_q_before)
    assert torch.equal(flat_q[-1:], expected_q_after)
    assert torch.equal(flat_k[:1], expected_k_before)
    assert torch.equal(flat_k[-1:], expected_k_after)
    assert torch.equal(guard_before, expected_guard)
    assert torch.equal(guard_after, expected_guard)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA graph capture is unavailable"
)
def test_olmo2_qk_norm_graph_output_reuse_across_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(33415)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    eps = 1e-6
    sentinel = 12345.0
    shape_q = (8, 512)
    shape_k = (8, 128)

    q_norm = RMSNorm(shape_q[-1], eps=eps).to(device=device, dtype=dtype)
    k_norm = RMSNorm(shape_k[-1], eps=eps).to(device=device, dtype=dtype)
    with torch.no_grad():
        q_norm.weight.copy_(torch.randn_like(q_norm.weight))
        k_norm.weight.copy_(torch.randn_like(k_norm.weight))
    attention = _make_attention(q_norm, k_norm)

    fresh_dispatch = []
    original_q_native = q_norm.forward_native
    original_k_native = k_norm.forward_native

    def wrapped_q_native(*args, **kwargs):
        fresh_dispatch.append("native")
        return original_q_native(*args, **kwargs)

    def wrapped_k_native(*args, **kwargs):
        fresh_dispatch.append("native")
        return original_k_native(*args, **kwargs)

    monkeypatch.setattr(q_norm, "forward_native", wrapped_q_native)
    monkeypatch.setattr(k_norm, "forward_native", wrapped_k_native)

    q_base = torch.arange(
        shape_q[0] * shape_q[1], device=device, dtype=torch.float32
    ).reshape(shape_q) / (shape_q[0] * shape_q[1])
    k_base = torch.arange(
        shape_k[0] * shape_k[1], device=device, dtype=torch.float32
    ).reshape(shape_k) / (shape_k[0] * shape_k[1])
    batches = [
        ((q_base + offset).to(dtype), (k_base + offset).to(dtype))
        for offset in range(3)
    ]

    flat_q, static_q = _padded_tensor(shape_q, dtype, device, sentinel)
    flat_k, static_k = _padded_tensor(shape_k, dtype, device, sentinel)
    static_q.copy_(batches[0][0])
    static_k.copy_(batches[0][1])

    original_backend = get_fused_op_backend()
    set_fused_op_backend(KernelBackend.TORCH)
    enable_fused_op_trace()
    clear_fused_op_trace()
    try:
        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream), torch.no_grad():
            attention._apply_qk_norm(static_q, static_k)
        torch.cuda.current_stream().wait_stream(warmup_stream)

        graph = torch.cuda.CUDAGraph()
        with (
            torch.no_grad(),
            model_capture_mode(),
            torch.cuda.graph(graph),
        ):
            guard_before = torch.full((4,), sentinel, dtype=dtype, device=device)
            captured_q, captured_k = attention._apply_qk_norm(static_q, static_k)
            guard_after = torch.full((4,), sentinel, dtype=dtype, device=device)
        capture_backends = [record.backend for record in get_fused_op_trace()]
        clear_fused_op_trace()
        assert capture_backends == ["torch", "torch"]

        captured_q_ptr = captured_q.data_ptr()
        captured_k_ptr = captured_k.data_ptr()
        assert captured_q.dtype is dtype
        assert captured_k.dtype is dtype
        assert captured_q_ptr != static_q.data_ptr()
        assert captured_k_ptr != static_k.data_ptr()

        fresh_outputs = []
        fresh_dispatch.clear()
        with torch.no_grad():
            for batch_q, batch_k in batches:
                fresh_q, fresh_k = attention._apply_qk_norm(batch_q, batch_k)
                fresh_outputs.append((fresh_q, fresh_k))
        assert fresh_dispatch == ["native"] * 6

        fresh_q_ptrs = [output[0].data_ptr() for output in fresh_outputs]
        fresh_k_ptrs = [output[1].data_ptr() for output in fresh_outputs]
        assert len(set(fresh_q_ptrs)) == len(fresh_q_ptrs)
        assert len(set(fresh_k_ptrs)) == len(fresh_k_ptrs)
        assert captured_q_ptr not in fresh_q_ptrs
        assert captured_k_ptr not in fresh_k_ptrs

        for index, (batch_q, batch_k) in enumerate(batches):
            static_q.copy_(batch_q)
            static_k.copy_(batch_k)
            graph.replay()
            torch.cuda.synchronize()
            _assert_sentinels(
                flat_q, flat_k, guard_before, guard_after, sentinel
            )
            assert captured_q.data_ptr() == captured_q_ptr
            assert captured_k.data_ptr() == captured_k_ptr
            assert captured_q.dtype is dtype
            assert captured_k.dtype is dtype

            q_reference = _reference(batch_q, q_norm.weight, eps)
            k_reference = _reference(batch_k, k_norm.weight, eps)
            torch.testing.assert_close(
                captured_q, q_reference, rtol=2e-2, atol=2e-2
            )
            torch.testing.assert_close(
                captured_k, k_reference, rtol=2e-2, atol=2e-2
            )
            torch.testing.assert_close(
                fresh_outputs[index][0], q_reference, rtol=2e-2, atol=2e-2
            )
            torch.testing.assert_close(
                fresh_outputs[index][1], k_reference, rtol=2e-2, atol=2e-2
            )
            assert torch.equal(captured_q, fresh_outputs[index][0])
            assert torch.equal(captured_k, fresh_outputs[index][1])

        assert not torch.equal(captured_q, fresh_outputs[0][0])
        assert not torch.equal(captured_k, fresh_outputs[0][1])

        with pytest.raises(NotImplementedError):
            q_norm.forward(batches[0][0], backend=KernelBackend.TRITON)
    finally:
        clear_fused_op_trace()
        disable_fused_op_trace()
        set_fused_op_backend(original_backend)
