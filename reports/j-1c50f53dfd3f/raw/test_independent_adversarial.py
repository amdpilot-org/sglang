from contextlib import nullcontext

import pytest
import torch

from test_mxfp4_empty_input import _ActivationType, _dispatch, _runner_and_quant_info
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.topk import StandardTopKOutput


@pytest.fixture
def boundary(monkeypatch):
    import sglang.srt.layers.moe.moe_runner.flashinfer_cutlass as module

    calls = []

    def kernel(**kwargs):
        calls.append(kwargs)
        kwargs["output"].fill_(7)
        return (kwargs["output"],)

    monkeypatch.setattr(module, "_flashinfer_cutlass_fused_moe", lambda: (kernel, _ActivationType))
    monkeypatch.setattr(module, "use_symmetric_memory", lambda *a, **k: nullcontext())
    monkeypatch.setattr(module, "get_tp_group", lambda: object())
    monkeypatch.setattr(module, "is_allocation_symmetric", lambda: False)
    monkeypatch.setattr(module.envs.SGLANG_FLASHINFER_MOE_FUSED_FINALIZE, "get", lambda: False)
    return calls


def test_empty_padded_contract_and_no_kernel(boundary):
    device = torch.device("cuda")
    runner, quant = _runner_and_quant_info(device)
    quant.padded_hidden = 32
    source = _dispatch(0, device)
    out = runner.run(source, quant).hidden_states
    assert boundary == []
    assert out.shape == (0, 16)
    assert out.dtype == torch.bfloat16
    assert out.device == source.hidden_states.device
    assert out.is_contiguous()


def test_nonempty_still_calls_boundary_with_expected_shapes(boundary):
    device = torch.device("cuda")
    runner, quant = _runner_and_quant_info(device)
    out = runner.run(_dispatch(2, device), quant).hidden_states
    assert len(boundary) == 1
    assert boundary[0]["input"].shape == (2, 16)
    assert boundary[0]["token_selected_experts"].shape == (2, 2)
    assert boundary[0]["output"].shape == (2, 16)
    assert torch.equal(out, torch.full_like(out, 7))


@pytest.mark.parametrize("which", ["hidden_rank", "ids_rank", "weights_rank", "shape_disagreement"])
def test_malformed_empty_structures_rejected_before_kernel(boundary, which):
    device = torch.device("cuda")
    runner, quant = _runner_and_quant_info(device)
    dispatch = _dispatch(0, device)
    hidden = dispatch.hidden_states
    ids = dispatch.topk_output.topk_ids
    weights = dispatch.topk_output.topk_weights
    if which == "hidden_rank":
        hidden = torch.empty((0, 1, 16), dtype=torch.bfloat16, device=device)
    elif which == "ids_rank":
        ids = torch.empty((0, 2, 1), dtype=torch.int32, device=device)
    elif which == "weights_rank":
        weights = torch.empty((0, 2, 1), device=device)
    else:
        weights = torch.empty((0, 1), device=device)
    dispatch = StandardDispatchOutput(
        hidden_states=hidden,
        hidden_states_scale=None,
        topk_output=StandardTopKOutput(
            topk_weights=weights,
            topk_ids=ids,
            router_logits=dispatch.topk_output.router_logits,
        ),
    )
    with pytest.raises(ValueError):
        runner.run(dispatch, quant)
    assert boundary == []
