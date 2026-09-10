import pytest
import torch

from sglang.kernels.fused_op import (
    clear_fused_op_trace,
    disable_fused_op_trace,
    enable_fused_op_trace,
    get_fused_op_trace,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.model_executor.runner import model_capture_mode
from sglang.srt.models.olmo2 import Olmo2Attention
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-large")
register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd")


def _reference(input, weight, eps):
    output_dtype = input.dtype
    input = input.float()
    return (
        input
        * (input.pow(2).mean(dim=-1, keepdim=True) + eps).rsqrt()
        * weight.float()
    ).to(output_dtype)


def _make_attention(q_norm, k_norm):
    attention = object.__new__(Olmo2Attention)
    torch.nn.Module.__init__(attention)
    attention.tp_size = 1
    attention.tp_rank = 0
    attention.alt_stream = torch.cuda.Stream()
    attention.q_norm = q_norm
    attention.k_norm = k_norm
    return attention


def _trace_backends():
    return [record.backend for record in get_fused_op_trace()]


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA graph capture is unavailable"
)
def test_olmo2_eager_qk_norm_uses_same_dispatch_as_capture():
    torch.manual_seed(6070)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    eps = 1e-6
    q = torch.randn(256, 1024, device=device, dtype=dtype)
    k = torch.randn(256, 256, device=device, dtype=dtype)
    q_norm = RMSNorm(1024, eps=eps).to(device=device, dtype=dtype)
    k_norm = RMSNorm(256, eps=eps).to(device=device, dtype=dtype)
    with torch.no_grad():
        q_norm.weight.copy_(torch.randn_like(q_norm.weight))
        k_norm.weight.copy_(torch.randn_like(k_norm.weight))

    attention = _make_attention(q_norm, k_norm)
    enable_fused_op_trace()
    clear_fused_op_trace()
    try:
        with torch.no_grad():
            eager_q, eager_k = attention._apply_qk_norm(q, k)
        torch.cuda.synchronize()
        eager_backends = _trace_backends()
        assert len(eager_backends) == 2
        assert eager_backends[0] == eager_backends[1]

        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream), torch.no_grad():
            attention._apply_qk_norm(q, k)
        torch.cuda.current_stream().wait_stream(warmup_stream)

        static_q, static_k = q.clone(), k.clone()
        graph = torch.cuda.CUDAGraph()
        clear_fused_op_trace()
        with (
            torch.no_grad(),
            model_capture_mode(),
            torch.cuda.graph(graph),
        ):
            captured_q, captured_k = attention._apply_qk_norm(static_q, static_k)
        capture_backends = _trace_backends()
        assert capture_backends == eager_backends

        static_q.copy_(q)
        static_k.copy_(k)
        graph.replay()
        torch.cuda.synchronize()
    finally:
        clear_fused_op_trace()
        disable_fused_op_trace()

    q_reference = _reference(q, q_norm.weight, eps)
    k_reference = _reference(k, k_norm.weight, eps)
    torch.testing.assert_close(eager_q, q_reference, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(eager_k, k_reference, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(captured_q, q_reference, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(captured_k, k_reference, rtol=2e-2, atol=2e-2)
