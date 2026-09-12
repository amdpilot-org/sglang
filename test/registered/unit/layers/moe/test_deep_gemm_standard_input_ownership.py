"""Ownership regression tests for the standard-to-DeepGEMM pre-permute path."""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.moe.moe_runner import deep_gemm
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.deep_gemm import DeepGemmMoeQuantInfo
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=5, stage="base-a", runner_config="1-gpu-small")
register_amd_ci(est_time=5, stage="stage-a", runner_config="1-gpu-small-amd")


@pytest.mark.parametrize("use_masked_layout", [False, True])
@pytest.mark.parametrize("inplace", [False, True])
def test_standard_pre_permute_respects_input_ownership(
    monkeypatch, use_masked_layout, inplace
):
    """Only an inplace runner may invalidate the caller's aliased activation."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hidden_states = torch.arange(16, device=device, dtype=torch.bfloat16).reshape(4, 4)
    caller_alias = hidden_states
    expected = hidden_states.clone()
    topk_ids = torch.tensor([[0], [1], [0], [1]], device=device, dtype=torch.int64)
    topk_weights = torch.ones((4, 1), device=device, dtype=torch.float32)
    dispatch_output = StandardDispatchOutput(
        hidden_states=hidden_states,
        hidden_states_scale=None,
        topk_output=(topk_weights, topk_ids, None),
    )
    quant_info = DeepGemmMoeQuantInfo(
        w13_weight=torch.empty((2, 1, 4), device=device, dtype=torch.bfloat16),
        w2_weight=torch.empty((2, 4, 1), device=device, dtype=torch.bfloat16),
        use_fp8=False,
    )
    runner_config = MoeRunnerConfig(
        num_experts=2,
        num_local_experts=2,
        hidden_size=4,
        top_k=1,
        inplace=inplace,
    )

    disposed = []

    def dispose(tensor):
        disposed.append(tensor)
        tensor.set_(torch.empty((0,), device=tensor.device, dtype=tensor.dtype))

    monkeypatch.setattr(deep_gemm, "dispose_tensor", dispose)
    monkeypatch.setattr(deep_gemm.deep_gemm_sm120, "is_supported", lambda: True)
    monkeypatch.setattr(
        deep_gemm.deep_gemm_sm120, "maybe_pre_permute", lambda *args: None
    )
    monkeypatch.setattr(
        deep_gemm.deep_gemm_sm120,
        "allows_masked_standard_layout",
        lambda: use_masked_layout,
    )
    monkeypatch.setattr(
        deep_gemm,
        "_should_use_masked_standard_layout",
        lambda *args: use_masked_layout,
    )

    import sglang.kernels.ops.moe.ep_moe_kernels as kernels

    def preprocess(*args, **kwargs):
        return (
            torch.tensor([2, 2], device=device, dtype=torch.int32),
            None,
            torch.arange(4, device=device, dtype=torch.int32).reshape(4, 1),
            expected.clone(),
            torch.ones((4, 1), device=device, dtype=torch.float32),
        )

    monkeypatch.setattr(kernels, "moe_ep_deepgemm_preprocess", preprocess)

    if not use_masked_layout:
        monkeypatch.setattr(
            deep_gemm.deep_gemm_wrapper,
            "get_contiguous_layout_alignment",
            lambda *args: 1,
        )
        monkeypatch.setattr(
            kernels,
            "fused_moe_dispatch_index",
            lambda *args, **kwargs: (
                torch.tensor([2, 2], device=device, dtype=torch.int32),
                torch.empty(1, device=device),
            ),
        )

        def scatter(source, _source_scale, *args, **kwargs):
            packed_input = args[4]
            src2dst = args[7]
            packed_input.copy_(source)
            src2dst.copy_(torch.arange(4, device=device).reshape(4, 1))

        monkeypatch.setattr(kernels, "ep_scatter", scatter)
        monkeypatch.setattr(
            deep_gemm, "get_exec", lambda: SimpleNamespace(deterministic=None)
        )

    result = deep_gemm.pre_permute_standard_to_deep_gemm(
        dispatch_output, quant_info, runner_config, {}
    )

    assert result.hidden_states.numel() == expected.numel()
    if inplace:
        assert caller_alias.numel() == 0
        assert any(tensor is caller_alias for tensor in disposed)
    else:
        assert caller_alias.shape == expected.shape
        torch.testing.assert_close(caller_alias, expected)
        assert all(tensor is not caller_alias for tensor in disposed)
