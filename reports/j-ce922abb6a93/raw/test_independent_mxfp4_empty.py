from contextlib import nullcontext
from enum import IntEnum

import pytest
import torch

from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.flashinfer_cutlass import FlashInferCutlassMxfp4MoeQuantInfo
from sglang.srt.layers.moe.moe_runner.runner import MoeRunner
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.moe.utils import MoeRunnerBackend


class ActivationType(IntEnum):
    Swiglu = 0
    SwigluStep = 1


def make_runner(device):
    config = MoeRunnerConfig(num_experts=2, num_local_experts=2, hidden_size=16,
                             intermediate_size_per_partition=32, top_k=2,
                             activation="silu", is_gated=True)
    runner = MoeRunner(MoeRunnerBackend.FLASHINFER_MXFP4, config)
    quant = FlashInferCutlassMxfp4MoeQuantInfo(
        w13_weight=torch.empty((2, 64, 8), dtype=torch.uint8, device=device),
        w2_weight=torch.empty((2, 16, 16), dtype=torch.uint8, device=device),
        w13_weight_scale=torch.empty((2, 64, 4), dtype=torch.uint8, device=device),
        w2_weight_scale=torch.empty((2, 16, 4), dtype=torch.uint8, device=device),
    )
    return runner, quant


def dispatch(device, xshape=(0, 16), route_shape=(0, 2)):
    return StandardDispatchOutput(
        hidden_states=torch.empty(xshape, dtype=torch.bfloat16, device=device),
        hidden_states_scale=None,
        topk_output=StandardTopKOutput(
            topk_weights=torch.empty(route_shape, dtype=torch.float32, device=device),
            topk_ids=torch.empty(route_shape, dtype=torch.int32, device=device),
            router_logits=torch.empty((route_shape[0], 2), device=device),
        ),
    )


@pytest.fixture(autouse=True)
def boundary(monkeypatch):
    import sglang.srt.layers.moe.moe_runner.flashinfer_cutlass as module
    calls = []
    def kernel(**kwargs):
        calls.append(kwargs)
        kwargs["output"].fill_(7)
        return (kwargs["output"],)
    monkeypatch.setattr(module, "_flashinfer_cutlass_fused_moe", lambda: (kernel, ActivationType))
    monkeypatch.setattr(module, "use_symmetric_memory", lambda *a, **kw: nullcontext())
    monkeypatch.setattr(module, "get_tp_group", lambda: object())
    monkeypatch.setattr(module, "is_allocation_symmetric", lambda: False)
    monkeypatch.setattr(module.envs.SGLANG_FLASHINFER_MOE_FUSED_FINALIZE, "get", lambda: False)
    return calls


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_valid_empty_and_nonempty_contract(boundary):
    device = torch.device("cuda")
    runner, quant = make_runner(device)
    result = runner.run(dispatch(device), quant).hidden_states
    assert tuple(result.shape) == (0, 16)
    assert result.dtype == torch.bfloat16
    assert result.device == dispatch(device).hidden_states.device
    assert boundary == []
    nonempty = dispatch(device, (1, 16), (1, 2))
    result = runner.run(nonempty, quant).hidden_states
    assert len(boundary) == 1
    torch.testing.assert_close(result, torch.full_like(result, 7))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("route_shape", [(0, 0), (0, 1), (0, 3)])
def test_empty_rejects_wrong_topk_width(boundary, route_shape):
    device = torch.device("cuda")
    runner, quant = make_runner(device)
    with pytest.raises((ValueError, AssertionError)):
        runner.run(dispatch(device, route_shape=route_shape), quant)
    assert boundary == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_empty_rejects_wrong_hidden_width(boundary):
    device = torch.device("cuda")
    runner, quant = make_runner(device)
    with pytest.raises((ValueError, AssertionError)):
        runner.run(dispatch(device, xshape=(0, 15)), quant)
    assert boundary == []
