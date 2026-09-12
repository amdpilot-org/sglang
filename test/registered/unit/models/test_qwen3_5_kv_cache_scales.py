import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers.attention.triton_backend import _get_kv_cache_descales
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.model_executor.model_runner_components.load_model_utils import (
    load_kv_cache_scales,
)
from sglang.srt.models.qwen3_5 import (
    Qwen3_5AttentionDecoderLayer,
    Qwen3_5ForCausalLM,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5LinearDecoderLayer,
    Qwen3_5MoeForCausalLM,
    Qwen3_5MoeForConditionalGeneration,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestQwen3_5KVCacheScales(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmpdir.cleanup()

    def _write_scales(
        self,
        *,
        model_type="qwen3_5_moe",
        scaling_factor=None,
        nested=False,
        dtype="float8_e4m3fn",
    ):
        if scaling_factor is None:
            scaling_factor = {"0": {"1": 0.20}, "1": {"1": 0.50}}
        payload = {"model_type": model_type}
        if nested:
            payload["kv_cache"] = {
                "dtype": dtype,
                "scaling_factor": scaling_factor,
            }
        else:
            payload["scaling_factor"] = scaling_factor
            if dtype != "float8_e4m3fn":
                payload["dtype"] = dtype
        path = (
            Path(self.tmpdir.name)
            / f"scales-{len(list(Path(self.tmpdir.name).iterdir()))}.json"
        )
        path.write_text(json.dumps(payload))
        return str(path)

    @staticmethod
    def _attention_layer():
        layer = Qwen3_5AttentionDecoderLayer.__new__(Qwen3_5AttentionDecoderLayer)
        torch.nn.Module.__init__(layer)
        layer.attn = SimpleNamespace(
            k_scale=None, v_scale=None, k_scale_float=None, v_scale_float=None
        )
        return layer

    @staticmethod
    def _linear_layer():
        layer = Qwen3_5LinearDecoderLayer.__new__(Qwen3_5LinearDecoderLayer)
        torch.nn.Module.__init__(layer)
        return layer

    def _model(self, cls, model_type="qwen3_5_moe_text"):
        model = cls.__new__(cls)
        torch.nn.Module.__init__(model)
        model.config = SimpleNamespace(
            model_type=model_type,
            num_hidden_layers=3,
            layers_block_type=["linear_attention", "attention", "linear_attention"],
        )
        model.layers = torch.nn.ModuleList(
            [self._linear_layer(), self._attention_layer(), PPMissingLayer()]
        )
        return model

    def _load(self, model, path, *, tp_size=2, tp_rank=1):
        with patch(
            "sglang.srt.models.qwen3_5.get_parallel",
            return_value=SimpleNamespace(tp_size=tp_size, tp_rank=tp_rank),
        ):
            model.load_kv_cache_scales(path)

    def test_flat_issue_schema_and_nested_schema_load_all_representations(self):
        for nested in (False, True):
            for cls in (Qwen3_5ForCausalLM, Qwen3_5MoeForCausalLM):
                with self.subTest(nested=nested, cls=cls.__name__):
                    model = self._model(cls)
                    self._load(model, self._write_scales(nested=nested))
                    attn = model.layers[1].attn
                    self.assertEqual(attn.k_scale.item(), 0.5)
                    self.assertEqual(attn.v_scale.item(), 0.5)
                    self.assertEqual(attn.k_scale_float, 0.5)
                    self.assertEqual(attn.v_scale_float, 0.5)
                    self.assertEqual(_get_kv_cache_descales(attn), (0.5, 0.5))

    def test_hybrid_layer_set_and_rank_are_still_validated(self):
        invalid_maps = (
            {"0": {"0": 0.2}, "1": {"0": 0.5}},
            {"0": {"1": 0.2}},
        )
        for scaling_factor in invalid_maps:
            with self.subTest(scaling_factor=scaling_factor):
                model = self._model(Qwen3_5MoeForCausalLM)
                with self.assertLogs(
                    "sglang.srt.model_loader.weight_utils", level="ERROR"
                ):
                    self._load(model, self._write_scales(scaling_factor=scaling_factor))
                self.assertIsNone(model.layers[1].attn.k_scale)

    def test_model_type_dtype_and_conflicting_layout_are_rejected(self):
        paths = [
            self._write_scales(model_type="qwen3_5"),
            self._write_scales(dtype="float8_e5m2"),
        ]
        conflicting = Path(self.tmpdir.name) / "conflicting.json"
        conflicting.write_text(
            json.dumps(
                {
                    "model_type": "qwen3_5_moe",
                    "scaling_factor": {"0": {"1": 0.2}, "1": {"1": 0.5}},
                    "kv_cache": {
                        "dtype": "float8_e4m3fn",
                        "scaling_factor": {"0": {"1": 0.2}, "1": {"1": 0.5}},
                    },
                }
            )
        )
        paths.append(str(conflicting))
        for path in paths:
            with self.subTest(path=path):
                model = self._model(Qwen3_5MoeForCausalLM)
                with self.assertLogs(
                    "sglang.srt.model_loader.weight_utils", level="ERROR"
                ):
                    self._load(model, path)
                self.assertIsNone(model.layers[1].attn.k_scale)

    def test_missing_and_malformed_calibration_do_not_mutate_scales(self):
        malformed = Path(self.tmpdir.name) / "malformed.json"
        malformed.write_text("{")
        for path in (str(Path(self.tmpdir.name) / "missing.json"), str(malformed)):
            with self.subTest(path=path):
                model = self._model(Qwen3_5MoeForCausalLM)
                with self.assertLogs(
                    "sglang.srt.model_loader.weight_utils", level="ERROR"
                ):
                    self._load(model, path)
                self.assertIsNone(model.layers[1].attn.k_scale)

    def test_dense_and_moe_conditional_wrappers_delegate(self):
        for cls in (
            Qwen3_5ForConditionalGeneration,
            Qwen3_5MoeForConditionalGeneration,
        ):
            with self.subTest(cls=cls.__name__):
                wrapper = cls.__new__(cls)
                torch.nn.Module.__init__(wrapper)
                wrapper.model = MagicMock()
                with patch(
                    "sglang.srt.model_executor.model_runner_components."
                    "load_model_utils.get_model",
                    return_value=SimpleNamespace(quantization_param_path="scales.json"),
                ):
                    load_kv_cache_scales(model=wrapper, kv_cache_dtype="fp8_e4m3")
                wrapper.model.load_kv_cache_scales.assert_called_once_with(
                    "scales.json"
                )


if __name__ == "__main__":
    unittest.main()
