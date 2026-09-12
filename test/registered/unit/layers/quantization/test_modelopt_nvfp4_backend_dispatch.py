import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang.srt.layers import moe
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.layers.quantization import modelopt_quant
from sglang.srt.layers.quantization.modelopt_quant import (
    ModelOptNvFp4FusedMoEMethod,
)


def _method(cached_backend=MoeRunnerBackend.FLASHINFER_CUTLASS):
    method = ModelOptNvFp4FusedMoEMethod.__new__(ModelOptNvFp4FusedMoEMethod)
    method.quant_config = SimpleNamespace(use_per_token_activation=False)
    method.moe_runner_config = MoeRunnerConfig(activation="silu")
    method._moe_runner_backend = cached_backend
    method.enable_flashinfer_trtllm_moe = False
    method.runner = SimpleNamespace(run=Mock(return_value="runner-output"))
    return method


def _cutlass_layer():
    tensor = torch.empty(0)
    return SimpleNamespace(
        w13_weight=tensor,
        w2_weight=tensor,
        w13_input_scale_quant=tensor,
        w13_blockscale_swizzled=tensor,
        g1_alphas=tensor,
        w2_input_scale_quant=tensor,
        w2_blockscale_swizzled=tensor,
        g2_alphas=tensor,
        moe_ep_size=1,
        moe_ep_rank=0,
        moe_tp_size=1,
        moe_tp_rank=0,
    )


def _install_quant_module(monkeypatch, name, class_name, **members):
    module = ModuleType(name)
    setattr(module, class_name, lambda **kwargs: SimpleNamespace(**kwargs))
    for member_name, member in members.items():
        setattr(module, member_name, member)
    monkeypatch.setitem(sys.modules, name, module)


@pytest.mark.parametrize("live_backend", [MoeRunnerBackend.TRITON, MoeRunnerBackend.AUTO])
def test_apply_uses_cached_cutlass_backend_across_global_state_changes(
    monkeypatch, live_backend
):
    """Exercise apply(), stubbing only the external runner boundary."""
    method = _method()
    monkeypatch.setattr(modelopt_quant, "get_moe_runner_backend", lambda: live_backend)
    monkeypatch.setattr(moe, "get_moe_runner_backend", lambda: live_backend)

    assert method.apply(_cutlass_layer(), object()) == "runner-output"
    quant_info = method.runner.run.call_args.args[1]
    assert quant_info.quant_type == "fp4"


def test_apply_without_cached_backend_uses_current_global(monkeypatch):
    method = _method()
    del method._moe_runner_backend
    monkeypatch.setattr(
        modelopt_quant,
        "get_moe_runner_backend",
        lambda: MoeRunnerBackend.FLASHINFER_CUTLASS,
    )
    monkeypatch.setattr(
        moe,
        "get_moe_runner_backend",
        lambda: MoeRunnerBackend.FLASHINFER_CUTLASS,
    )

    assert method.apply(_cutlass_layer(), object()) == "runner-output"


def test_draft_layer_keeps_cached_unsupported_backend_on_target_state_exit(monkeypatch):
    method = _method(MoeRunnerBackend.TRITON)
    monkeypatch.setattr(
        modelopt_quant,
        "get_moe_runner_backend",
        lambda: MoeRunnerBackend.FLASHINFER_CUTLASS,
    )

    with pytest.raises(NotImplementedError, match="MoeRunnerBackend.TRITON"):
        method.apply(_cutlass_layer(), object())


@pytest.mark.parametrize(
    "cached_backend, module_name, class_name",
    [
        (
            MoeRunnerBackend.FLASHINFER_TRTLLM,
            "sglang.srt.layers.moe.moe_runner.flashinfer_trtllm",
            "FlashInferTrtllmFp4MoeQuantInfo",
        ),
        (
            MoeRunnerBackend.FLASHINFER_CUTEDSL,
            "sglang.srt.layers.moe.moe_runner.flashinfer_cutedsl",
            "CuteDslFp4MoeQuantInfo",
        ),
        (
            MoeRunnerBackend.FLASHINFER_MEGAMOE,
            "sglang.srt.layers.moe.flashinfer_megamoe",
            "FlashInferMegaMoeQuantInfo",
        ),
    ],
)
def test_apply_uses_cached_backend_for_other_supported_branches(
    monkeypatch, cached_backend, module_name, class_name
):
    tensor = torch.empty(0)
    layer = _cutlass_layer()
    layer.w13_weight_scale = tensor
    layer.w2_weight_scale = tensor
    layer.g1_scale_c = tensor
    layer.num_experts = 1
    layer.num_local_experts = 1
    layer.intermediate_size_per_partition = 1
    layer._flashinfer_megamoe_forward = Mock()
    layer.should_fuse_routed_scaling_factor_in_topk = False

    extra = {}
    if cached_backend.is_flashinfer_cutedsl():
        monkeypatch.setattr(
            modelopt_quant, "is_flashinfer_cutedsl_v1_path", lambda: True
        )
        extra["ensure_cutedsl_wrapper"] = Mock()
    elif cached_backend.is_flashinfer_megamoe():
        extra["ensure_nvfp4_moe_layer_for_flashinfer_megamoe"] = lambda layer: "mega"
    _install_quant_module(monkeypatch, module_name, class_name, **extra)

    method = _method(cached_backend)
    monkeypatch.setattr(
        modelopt_quant, "get_moe_runner_backend", lambda: MoeRunnerBackend.TRITON
    )

    assert method.apply(layer, object()) == "runner-output"


def test_apply_uses_cached_marlin_backend(monkeypatch):
    method = _method(MoeRunnerBackend.MARLIN)
    method.get_marlin_quant_info = Mock(return_value="marlin-info")
    monkeypatch.setattr(
        modelopt_quant, "get_moe_runner_backend", lambda: MoeRunnerBackend.TRITON
    )

    assert method.apply(SimpleNamespace(), object()) == "runner-output"
    method.get_marlin_quant_info.assert_called_once()
