from pathlib import Path
from unittest.mock import Mock

import pytest

from sglang.srt.models.deepseek_common.attention_backend_handler import (
    AttnForwardMethod,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=12, suite="base-a-test-cpu")
from sglang.srt.models.deepseek_common.hardware_backend import (
    DeepseekV2AMDAttentionMixin,
    DeepseekV2CPUAttentionMixin,
)


def test_hardware_backend_layout_is_complete():
    backend = Path("python/sglang/srt/models/deepseek_common/hardware_backend")
    assert (backend / "deepseek_v2_amd_mixin.py").is_file()
    assert (backend / "deepseek_v2_cpu_mixin.py").is_file()
    model_source = Path("python/sglang/srt/models/deepseek_v2.py").read_text()
    assert "forward_amd_prepare(" in model_source
    assert "forward_cpu_prepare(" in model_source


@pytest.mark.parametrize(
    ("method", "prepare_name", "prepare_args", "core_name"),
    [
        (
            AttnForwardMethod.MHA_ROCM,
            "forward_normal_rocm_prepare",
            ("p", "h", "b", "z"),
            "forward_normal_core",
        ),
        (
            AttnForwardMethod.MHA_ONE_SHOT_ROCM,
            "forward_normal_one_shot_rocm_prepare",
            ("p", "h", "b", "z"),
            "forward_normal_core",
        ),
        (
            AttnForwardMethod.MHA_CHUNKED_KV_ROCM,
            "forward_normal_chunked_kv_rocm_prepare",
            ("p", "h", "b", "z"),
            "forward_normal_core",
        ),
        (
            AttnForwardMethod.MLA_ROCM,
            "forward_absorb_rocm_prepare",
            ("p", "h", "b", "z", "scale", "topk"),
            "forward_absorb_rocm_core",
        ),
        (
            AttnForwardMethod.MLA_FUSED_ROPE_ROCM,
            "forward_absorb_fused_mla_rope_prepare",
            ("p", "h", "b", "z"),
            "forward_absorb_fused_mla_rope_core",
        ),
    ],
)
def test_amd_attention_dispatch(method, prepare_name, prepare_args, core_name):
    attention = Mock(spec=DeepseekV2AMDAttentionMixin)
    setattr(attention, prepare_name, Mock(return_value=("state",)))
    setattr(attention, core_name, Mock(return_value="output"))

    state = DeepseekV2AMDAttentionMixin.forward_amd_prepare(
        attention, method, "p", "h", "b", "z", "scale", "topk"
    )
    getattr(attention, prepare_name).assert_called_once_with(*prepare_args)
    assert (
        DeepseekV2AMDAttentionMixin.forward_amd_core(attention, method, state)
        == "output"
    )
    getattr(attention, core_name).assert_called_once_with("state")


def test_cpu_attention_dispatch():
    attention = Mock(spec=DeepseekV2CPUAttentionMixin)
    attention.forward_absorb_fused_mla_rope_cpu_prepare.return_value = ("state",)
    attention.forward_absorb_fused_mla_rope_cpu_core.return_value = "output"

    state = DeepseekV2CPUAttentionMixin.forward_cpu_prepare(
        attention, AttnForwardMethod.MLA_FUSED_ROPE_CPU, "p", "h", "b", "z"
    )
    attention.forward_absorb_fused_mla_rope_cpu_prepare.assert_called_once_with(
        "p", "h", "b", "z"
    )
    assert (
        DeepseekV2CPUAttentionMixin.forward_cpu_core(
            attention, AttnForwardMethod.MLA_FUSED_ROPE_CPU, state
        )
        == "output"
    )
    attention.forward_absorb_fused_mla_rope_cpu_core.assert_called_once_with("state")


def test_hardware_mixins_reject_foreign_methods():
    with pytest.raises(NotImplementedError):
        DeepseekV2AMDAttentionMixin.forward_amd_prepare(
            Mock(), AttnForwardMethod.MLA, None, None, None, None, None, None
        )
    with pytest.raises(NotImplementedError):
        DeepseekV2CPUAttentionMixin.forward_cpu_prepare(
            Mock(), AttnForwardMethod.MLA, None, None, None, None
        )
