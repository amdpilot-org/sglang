from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn
from transformers.configuration_utils import logger as transformers_config_logger

import sglang.srt.batch_overlap.two_batch_overlap as tbo
import sglang.srt.models.dots3_common.modeling as modeling
from sglang.srt.configs.dots3 import Dots3Config
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


class _Gate(nn.Module):
    def __init__(self):
        super().__init__()
        self.e_score_correction_bias = None


class _Experts(nn.Module):
    should_fuse_routed_scaling_factor_in_topk = False


class _CaptureDispatcher(nn.Module):
    calls = []

    def __init__(self, **kwargs):
        super().__init__()
        self.calls.append(kwargs)


@pytest.mark.parametrize("dtype", [None, torch.float16, torch.bfloat16, torch.float32])
def test_dots3_dispatcher_uses_config_dtype_without_deprecation(dtype):
    _CaptureDispatcher.calls.clear()
    config_kwargs = {} if dtype is None else {"torch_dtype": dtype}
    config = Dots3Config(n_shared_experts=None, **config_kwargs)
    backend = SimpleNamespace(
        is_deepep=lambda: True,
        is_mooncake=lambda: False,
        is_mori=lambda: False,
        is_nixl=lambda: False,
        is_pplx=lambda: False,
    )

    with (
        patch.object(
            modeling,
            "get_parallel",
            return_value=SimpleNamespace(tp_size=1, moe_ep_size=1),
        ),
        patch.object(
            modeling,
            "get_exec",
            return_value=SimpleNamespace(
                moe=SimpleNamespace(ep_num_redundant_experts=0)
            ),
        ),
        patch.object(modeling, "get_moe_a2a_backend", return_value=backend),
        patch.object(tbo, "get_moe_a2a_backend", return_value=backend),
        patch.object(tbo, "is_tbo_enabled", return_value=False),
        patch.object(tbo, "DeepEPDispatcher", _CaptureDispatcher),
        patch.object(
            modeling.parallel_state,
            "get_tp_group",
            return_value=SimpleNamespace(device_group=object()),
        ),
        patch.object(modeling, "Dots3MoEGate", return_value=_Gate()),
        patch.object(
            modeling, "get_moe_impl_class", return_value=lambda **_: _Experts()
        ),
        patch.object(modeling, "TopK", return_value=nn.Identity()),
        patch.object(modeling, "is_shared_experts_fusion_disabled", return_value=True),
        patch.object(modeling, "get_deepep_mode", return_value="normal"),
        patch.object(transformers_config_logger, "warning_once") as warning_once,
    ):
        moe = modeling.Dots3MoE(config, layer_id=0)

    assert isinstance(moe.deepep_dispatcher, tbo.MaybeTboDeepEPDispatcher)
    assert len(_CaptureDispatcher.calls) == 1
    assert _CaptureDispatcher.calls[0]["params_dtype"] is config.dtype
    warning_once.assert_not_called()
