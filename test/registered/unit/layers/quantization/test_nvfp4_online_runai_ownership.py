import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.quantization.nvfp4_online import (
    ModelOptNvFp4OnlineFusedMoEMethod,
)
from sglang.srt.model_loader.weight_utils import RUNAI_STREAMER_TENSOR_ATTR
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestNvfp4OnlineRunaiOwnership(CustomTestCase):
    def _make_loader(self, *, gated=False, fp8_dequantizer=None):
        layer = SimpleNamespace(
            moe_runner_config=SimpleNamespace(is_gated=gated),
            w13_weight_scale=torch.nn.Parameter(torch.empty(1)),
            w2_weight_scale=torch.nn.Parameter(torch.empty(1)),
            w13_weight_scale_2=torch.nn.Parameter(torch.empty(1)),
            w2_weight_scale_2=torch.nn.Parameter(torch.empty(1)),
            _map_global_expert_id_to_local_expert_id=lambda _expert_id: 0,
        )
        return ModelOptNvFp4OnlineFusedMoEMethod.get_online_weight_loader(
            layer,
            lambda *args, **kwargs: None,
            layer_log_name="test layer",
            fp8_dequantizer=fp8_dequantizer,
        )

    @staticmethod
    def _streamed_view(buffer, value, dtype):
        buffer.fill_(value)
        view = (
            buffer.to(dtype).view(2, 2) if buffer.dtype != dtype else buffer.view(2, 2)
        )
        setattr(view, RUNAI_STREAMER_TENSOR_ATTR, True)
        return view

    def test_pending_fp8_weight_owns_runai_tensor(self):
        seen = []

        def dequantize(weight, scale, device):
            seen.append(weight.float().clone())
            return weight.float()

        loader = self._make_loader(fp8_dequantizer=dequantize)
        param = torch.nn.Parameter(torch.empty(1))
        staging = torch.full((2, 2), 1.0, dtype=torch.float8_e4m3fn)
        weight = self._streamed_view(staging, 1.0, torch.float8_e4m3fn)
        loader(param, weight, "experts.0.w2.weight", "w2", 0)
        staging.fill_(2.0)

        with patch.object(
            ModelOptNvFp4OnlineFusedMoEMethod,
            "_quantize_weight_nvfp4",
            return_value=(torch.empty(1), torch.empty(1), torch.empty(1)),
        ):
            loader(param, torch.tensor(0.5), "experts.0.w2.weight_scale", "w2", 0)

        torch.testing.assert_close(seen[0], torch.ones((2, 2)))

    def test_pending_fp8_scale_owns_runai_tensor(self):
        seen = []

        def dequantize(weight, scale, device):
            seen.append(scale.clone())
            return weight.float()

        loader = self._make_loader(fp8_dequantizer=dequantize)
        param = torch.nn.Parameter(torch.empty(1))
        staging = torch.full((2, 2), 0.5)
        scale = self._streamed_view(staging, 0.5, torch.float32)
        loader(param, scale, "experts.0.w2.weight_scale", "w2", 0)
        staging.fill_(3.0)

        with patch.object(
            ModelOptNvFp4OnlineFusedMoEMethod,
            "_quantize_weight_nvfp4",
            return_value=(torch.empty(1), torch.empty(1), torch.empty(1)),
        ):
            loader(
                param,
                torch.ones((2, 2), dtype=torch.float8_e4m3fn),
                "experts.0.w2.weight",
                "w2",
                0,
            )

        torch.testing.assert_close(seen[0], torch.full((2, 2), 0.5))

    def test_pending_shared_expert_fp8_weight_owns_runai_tensor(self):
        seen = []

        def dequantize(weight, scale, device):
            seen.append(weight.float().clone())
            return weight.float()

        loader = self._make_loader(fp8_dequantizer=dequantize)
        param = torch.nn.Parameter(torch.empty(1))
        staging = torch.full((2, 2), 1.0, dtype=torch.float8_e4m3fn)
        weight = self._streamed_view(staging, 1.0, torch.float8_e4m3fn)
        loader(param, weight, "shared_expert.w2.weight", "w2", None)
        staging.fill_(2.0)

        with patch.object(
            ModelOptNvFp4OnlineFusedMoEMethod,
            "_quantize_weight_nvfp4",
            return_value=(torch.empty(1), torch.empty(1), torch.empty(1)),
        ):
            loader(
                param,
                torch.tensor(0.5),
                "shared_expert.w2.weight_scale",
                "w2",
                None,
            )

        torch.testing.assert_close(seen[0], torch.ones((2, 2)))

    def test_pending_gated_weight_owns_runai_tensor(self):
        seen = []

        def quantize(weight):
            seen.append(weight.clone())
            rows = weight.shape[0]
            return torch.empty(rows, 1), torch.empty(rows, 1), torch.empty(1)

        loader = self._make_loader(gated=True)
        param = torch.nn.Parameter(torch.empty(1))
        staging = torch.full((2, 2), 1.0)
        w1 = self._streamed_view(staging, 1.0, torch.float32)
        loader(param, w1, "experts.0.w1.weight", "w1", 0)
        staging.fill_(4.0)

        with patch.object(
            ModelOptNvFp4OnlineFusedMoEMethod,
            "_quantize_weight_nvfp4",
            side_effect=quantize,
        ):
            loader(param, torch.full((2, 2), 2.0), "experts.0.w3.weight", "w3", 0)

        torch.testing.assert_close(
            seen[0], torch.tensor([[1.0, 1.0], [1.0, 1.0], [2.0, 2.0], [2.0, 2.0]])
        )

    def test_unmarked_pending_weight_is_not_copied(self):
        seen = []

        def quantize(weight):
            seen.append(weight.clone())
            rows = weight.shape[0]
            return torch.empty(rows, 1), torch.empty(rows, 1), torch.empty(1)

        loader = self._make_loader(gated=True)
        param = torch.nn.Parameter(torch.empty(1))
        ordinary = torch.ones((2, 2))
        loader(param, ordinary, "experts.0.w1.weight", "w1", 0)
        ordinary.fill_(4.0)

        with patch.object(
            ModelOptNvFp4OnlineFusedMoEMethod,
            "_quantize_weight_nvfp4",
            side_effect=quantize,
        ):
            loader(param, torch.full((2, 2), 2.0), "experts.0.w3.weight", "w3", 0)

        torch.testing.assert_close(seen[0][:2], torch.full((2, 2), 4.0))


if __name__ == "__main__":
    unittest.main()
