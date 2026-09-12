from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.layers.quantization.nvfp4_online import (
    ModelOptNvFp4OnlineFusedMoEMethod,
)
from sglang.srt.model_loader.weight_utils import RUNAI_STREAMER_TENSOR_ATTR


def make_loader(*, gated=False, dequantizer=None):
    layer = SimpleNamespace(
        moe_runner_config=SimpleNamespace(is_gated=gated),
        w13_weight_scale=torch.nn.Parameter(torch.empty(1)),
        w2_weight_scale=torch.nn.Parameter(torch.empty(1)),
        w13_weight_scale_2=torch.nn.Parameter(torch.empty(1)),
        w2_weight_scale_2=torch.nn.Parameter(torch.empty(1)),
        _map_global_expert_id_to_local_expert_id=lambda expert_id: expert_id,
    )
    return ModelOptNvFp4OnlineFusedMoEMethod.get_online_weight_loader(
        layer,
        lambda *args, **kwargs: None,
        layer_log_name="independent review",
        fp8_dequantizer=dequantizer,
    )


def marked_view(buffer):
    view = buffer.view(2, 2)
    setattr(view, RUNAI_STREAMER_TENSOR_ATTR, True)
    return view


def test_two_experts_survive_repeated_reuse_of_one_fp8_staging_buffer():
    observed = {}

    def dequantize(weight, scale, device):
        observed[float(scale)] = weight.float().clone()
        return weight.float()

    loader = make_loader(dequantizer=dequantize)
    param = torch.nn.Parameter(torch.empty(1))
    staging = torch.empty((2, 2), dtype=torch.float8_e4m3fn)

    staging.fill_(1.0)
    loader(param, marked_view(staging), "experts.0.w2.weight", "w2", 0)
    staging.fill_(2.0)
    loader(param, marked_view(staging), "experts.1.w2.weight", "w2", 1)
    staging.fill_(9.0)

    fake = (torch.empty(1), torch.empty(1), torch.empty(1))
    with patch.object(
        ModelOptNvFp4OnlineFusedMoEMethod,
        "_quantize_weight_nvfp4",
        return_value=fake,
    ):
        loader(param, torch.tensor(0.25), "experts.1.w2.weight_scale", "w2", 1)
        loader(param, torch.tensor(0.5), "experts.0.w2.weight_scale", "w2", 0)

    torch.testing.assert_close(observed[0.25], torch.full((2, 2), 2.0))
    torch.testing.assert_close(observed[0.5], torch.full((2, 2), 1.0))


def test_reverse_gated_order_preserves_marked_w3():
    seen = []

    def quantize(weight):
        seen.append(weight.clone())
        rows = weight.shape[0]
        return torch.empty(rows, 1), torch.empty(rows, 1), torch.empty(1)

    loader = make_loader(gated=True)
    param = torch.nn.Parameter(torch.empty(1))
    staging = torch.full((2, 2), 3.0)
    loader(param, marked_view(staging), "experts.4.w3.weight", "w3", 4)
    staging.fill_(8.0)

    with patch.object(
        ModelOptNvFp4OnlineFusedMoEMethod,
        "_quantize_weight_nvfp4",
        side_effect=quantize,
    ):
        loader(param, torch.full((2, 2), 1.0), "experts.4.w1.weight", "w1", 4)

    torch.testing.assert_close(seen[0][:2], torch.full((2, 2), 3.0))
    torch.testing.assert_close(seen[0][2:], torch.full((2, 2), 1.0))


def test_false_marker_retains_existing_reference_semantics():
    seen = []

    def quantize(weight):
        seen.append(weight.clone())
        rows = weight.shape[0]
        return torch.empty(rows, 1), torch.empty(rows, 1), torch.empty(1)

    loader = make_loader(gated=True)
    param = torch.nn.Parameter(torch.empty(1))
    ordinary = torch.ones((2, 2))
    setattr(ordinary, RUNAI_STREAMER_TENSOR_ATTR, False)
    loader(param, ordinary, "experts.2.w1.weight", "w1", 2)
    ordinary.fill_(7.0)

    with patch.object(
        ModelOptNvFp4OnlineFusedMoEMethod,
        "_quantize_weight_nvfp4",
        side_effect=quantize,
    ):
        loader(param, torch.full((2, 2), 2.0), "experts.2.w3.weight", "w3", 2)

    torch.testing.assert_close(seen[0][:2], torch.full((2, 2), 7.0))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU unavailable")
def test_marked_fp8_weight_owns_actual_gpu_staging_storage():
    seen = []

    def dequantize(weight, scale, device):
        assert weight.device.type == "cuda"
        seen.append(weight.float().cpu())
        return weight.float()

    loader = make_loader(dequantizer=dequantize)
    param = torch.nn.Parameter(torch.empty(1, device="cuda"))
    staging = torch.full((2, 2), 1.0, dtype=torch.float8_e4m3fn, device="cuda")
    loader(param, marked_view(staging), "experts.3.w2.weight", "w2", 3)
    staging.fill_(6.0)

    fake = (
        torch.empty(1, device="cuda"),
        torch.empty(1, device="cuda"),
        torch.empty(1, device="cuda"),
    )
    with patch.object(
        ModelOptNvFp4OnlineFusedMoEMethod,
        "_quantize_weight_nvfp4",
        return_value=fake,
    ):
        loader(
            param,
            torch.tensor(0.5, device="cuda"),
            "experts.3.w2.weight_scale",
            "w2",
            3,
        )

    torch.testing.assert_close(seen[0], torch.ones((2, 2)))
