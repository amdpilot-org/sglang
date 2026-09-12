from contextlib import contextmanager
from enum import IntEnum

import pytest
import torch

from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.flashinfer_cutlass import (
    FlashInferCutlassMxfp4MoeQuantInfo,
)
from sglang.srt.layers.moe.moe_runner.runner import MoeRunner
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-large")
register_amd_ci(est_time=20, suite="stage-b-test-1-gpu-small-amd")


class _ActivationType(IntEnum):
    Swiglu = 0
    SwigluStep = 1


def _runner_and_quant_info(device):
    runner = MoeRunner(
        MoeRunnerBackend.FLASHINFER_MXFP4,
        MoeRunnerConfig(
            num_experts=2,
            num_local_experts=2,
            hidden_size=16,
            intermediate_size_per_partition=32,
            top_k=2,
            activation="silu",
            is_gated=True,
        ),
    )
    quant_info = FlashInferCutlassMxfp4MoeQuantInfo(
        w13_weight=torch.empty((2, 64, 8), dtype=torch.uint8, device=device),
        w2_weight=torch.empty((2, 16, 16), dtype=torch.uint8, device=device),
        w13_weight_scale=torch.empty((2, 64, 4), dtype=torch.uint8, device=device),
        w2_weight_scale=torch.empty((2, 16, 4), dtype=torch.uint8, device=device),
    )
    return runner, quant_info


def _dispatch(tokens, device, *, hidden_size=16, topk_tokens=None, top_k=2):
    if topk_tokens is None:
        topk_tokens = tokens
    return StandardDispatchOutput(
        hidden_states=torch.randn(
            (tokens, hidden_size), dtype=torch.bfloat16, device=device
        ),
        hidden_states_scale=None,
        topk_output=StandardTopKOutput(
            topk_weights=torch.ones((topk_tokens, top_k), device=device),
            topk_ids=torch.zeros(
                (topk_tokens, top_k), dtype=torch.int32, device=device
            ),
            router_logits=torch.empty((topk_tokens, 2), device=device),
        ),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires one GPU")
def test_flashinfer_mxfp4_empty_input_stops_at_external_kernel_boundary(monkeypatch):
    import sglang.srt.layers.moe.moe_runner.flashinfer_cutlass as module

    calls = []
    symmetric_allocations = []

    def instrumented_kernel(**kwargs):
        calls.append(kwargs)
        if kwargs["input"].shape[0] == 0:
            raise RuntimeError("external kernel requires nonempty input")
        kwargs["output"].fill_(3)
        return (kwargs["output"],)

    @contextmanager
    def instrumented_symmetric_memory(*args, **kwargs):
        symmetric_allocations.append((args, kwargs))
        yield

    monkeypatch.setattr(
        module,
        "_flashinfer_cutlass_fused_moe",
        lambda: (instrumented_kernel, _ActivationType),
    )
    monkeypatch.setattr(module, "use_symmetric_memory", instrumented_symmetric_memory)
    monkeypatch.setattr(module, "get_tp_group", lambda: object())
    monkeypatch.setattr(module, "is_allocation_symmetric", lambda: False)
    monkeypatch.setattr(
        module.envs.SGLANG_FLASHINFER_MOE_FUSED_FINALIZE, "get", lambda: False
    )

    device = torch.device("cuda")
    runner, quant_info = _runner_and_quant_info(device)

    dispatch = _dispatch(0, device)
    empty = runner.run(dispatch, quant_info).hidden_states
    assert empty.shape == (0, 16)
    assert empty.dtype == torch.bfloat16
    assert empty.device == dispatch.hidden_states.device
    assert calls == []
    assert len(symmetric_allocations) == 1

    nonempty = runner.run(_dispatch(1, device), quant_info).hidden_states
    assert len(calls) == 1
    assert torch.equal(nonempty, torch.full_like(nonempty, 3))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires one GPU")
def test_flashinfer_mxfp4_does_not_accept_malformed_empty_routing(monkeypatch):
    import sglang.srt.layers.moe.moe_runner.flashinfer_cutlass as module

    monkeypatch.setattr(
        module,
        "_flashinfer_cutlass_fused_moe",
        lambda: (
            lambda **kwargs: (_ for _ in ()).throw(AssertionError()),
            _ActivationType,
        ),
    )
    monkeypatch.setattr(
        module,
        "use_symmetric_memory",
        lambda *a, **kw: __import__("contextlib").nullcontext(),
    )
    monkeypatch.setattr(module, "get_tp_group", lambda: object())
    monkeypatch.setattr(module, "is_allocation_symmetric", lambda: False)

    device = torch.device("cuda")
    runner, quant_info = _runner_and_quant_info(device)
    with pytest.raises(ValueError, match="token dimension"):
        runner.run(_dispatch(0, device, topk_tokens=1), quant_info)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires one GPU")
@pytest.mark.parametrize("top_k", [0, 1, 3])
def test_flashinfer_mxfp4_rejects_wrong_empty_routing_width(monkeypatch, top_k):
    import sglang.srt.layers.moe.moe_runner.flashinfer_cutlass as module

    monkeypatch.setattr(
        module,
        "_flashinfer_cutlass_fused_moe",
        lambda: (
            lambda **kwargs: (_ for _ in ()).throw(AssertionError()),
            _ActivationType,
        ),
    )

    device = torch.device("cuda")
    runner, quant_info = _runner_and_quant_info(device)
    with pytest.raises(ValueError, match="configured top_k 2"):
        runner.run(_dispatch(0, device, top_k=top_k), quant_info)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires one GPU")
def test_flashinfer_mxfp4_rejects_wrong_empty_hidden_width(monkeypatch):
    import sglang.srt.layers.moe.moe_runner.flashinfer_cutlass as module

    monkeypatch.setattr(
        module,
        "_flashinfer_cutlass_fused_moe",
        lambda: (
            lambda **kwargs: (_ for _ in ()).throw(AssertionError()),
            _ActivationType,
        ),
    )

    device = torch.device("cuda")
    runner, quant_info = _runner_and_quant_info(device)
    with pytest.raises(ValueError, match="configured hidden_size 16"):
        runner.run(_dispatch(0, device, hidden_size=15), quant_info)
