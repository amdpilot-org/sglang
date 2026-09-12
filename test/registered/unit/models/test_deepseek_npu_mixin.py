from unittest.mock import Mock, patch

import pytest

from sglang.srt.models.deepseek_common.attention_backend_handler import (
    AttnForwardMethod,
)
from sglang.srt.models.deepseek_common.hardware_backend import deepseek_v2_npu_mixin
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Attention(deepseek_v2_npu_mixin.DeepseekV2NPUAttentionMixin):
    pass


@pytest.mark.parametrize(
    ("method", "expects_topk"),
    [
        (AttnForwardMethod.MHA_NPU, False),
        (AttnForwardMethod.MLA_NPU, False),
        (AttnForwardMethod.DSA_NPU, True),
    ],
)
def test_npu_prepare_dispatches_with_dsa_topk_only(method, expects_topk):
    attention = _Attention()
    prepare = Mock(return_value="prepared")
    core = Mock()
    functions = {method: (prepare, core)}

    with patch.object(
        deepseek_v2_npu_mixin,
        "_npu_attention_functions",
        return_value=functions,
    ):
        result = attention.forward_npu_prepare(
            method, "positions", "hidden", "batch", "allocator", "scatter", "topk"
        )

    assert result == "prepared"
    assert prepare.call_args.args[0] is attention
    assert prepare.call_args.args[-1] == ("topk" if expects_topk else "scatter")
    assert len(prepare.call_args.args) == (7 if expects_topk else 6)


@pytest.mark.parametrize(
    "method",
    [
        AttnForwardMethod.MHA_NPU,
        AttnForwardMethod.MLA_NPU,
        AttnForwardMethod.DSA_NPU,
    ],
)
def test_npu_core_dispatches_inner_state(method):
    attention = _Attention()
    prepare = Mock()
    core = Mock(return_value="result")

    with patch.object(
        deepseek_v2_npu_mixin,
        "_npu_attention_functions",
        return_value={method: (prepare, core)},
    ):
        result = attention.forward_npu_core(method, ("one", "two"))

    assert result == "result"
    core.assert_called_once_with(attention, "one", "two")
