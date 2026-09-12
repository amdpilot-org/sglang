import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers.attention.triton_backend import _get_kv_cache_descales
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.models.qwen3_5 import (
    Qwen3_5AttentionDecoderLayer,
    Qwen3_5ForCausalLM,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5LinearDecoderLayer,
    Qwen3_5MoeForCausalLM,
    Qwen3_5MoeForConditionalGeneration,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestQwen3_5KVCacheScales(CustomTestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmpdir.cleanup()

    def _write_scales(self, *, model_type="qwen3_5_moe", scaling_factor=None):
        if scaling_factor is None:
            scaling_factor = {
                "0": {"1": 0.20},
                "1": {"1": 0.50},
            }
        path = Path(self.tmpdir.name) / "scales.json"
        path.write_text(
            json.dumps(
                {
                    "model_type": model_type,
                    "kv_cache": {
                        "dtype": "float8_e4m3fn",
                        "scaling_factor": scaling_factor,
                    },
                }
            )
        )
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

    def test_dense_and_moe_causal_classes_load_rank_specific_scales(self):
        path = self._write_scales()
        parallel = SimpleNamespace(tp_size=2, tp_rank=1)
        for cls in (Qwen3_5ForCausalLM, Qwen3_5MoeForCausalLM):
            with self.subTest(cls=cls.__name__):
                model = self._model(cls)
                with patch(
                    "sglang.srt.models.qwen3_5.get_parallel", return_value=parallel
                ):
                    model.load_kv_cache_scales(path)

                attn = model.layers[1].attn
                self.assertEqual(attn.k_scale.item(), 0.5)
                self.assertEqual(attn.v_scale.item(), 0.5)
                self.assertEqual(attn.k_scale_float, 0.5)
                self.assertEqual(attn.v_scale_float, 0.5)

    def test_loaded_float_scales_reach_the_triton_consumer(self):
        model = self._model(Qwen3_5MoeForCausalLM)
        with patch(
            "sglang.srt.models.qwen3_5.get_parallel",
            return_value=SimpleNamespace(tp_size=2, tp_rank=1),
        ):
            model.load_kv_cache_scales(self._write_scales())

        self.assertEqual(_get_kv_cache_descales(model.layers[1].attn), (0.5, 0.5))
        self.assertEqual(
            _get_kv_cache_descales(SimpleNamespace(k_scale=None, v_scale=None)),
            (1.0, 1.0),
        )

    def test_external_model_type_is_validated(self):
        model = self._model(Qwen3_5MoeForCausalLM)
        path = self._write_scales(model_type="qwen3_5")
        with (
            patch(
                "sglang.srt.models.qwen3_5.get_parallel",
                return_value=SimpleNamespace(tp_size=2, tp_rank=0),
            ),
            self.assertLogs("sglang.srt.model_loader.weight_utils", level="WARNING"),
        ):
            model.load_kv_cache_scales(path)
        self.assertIsNone(model.layers[1].attn.k_scale)

    def test_missing_and_malformed_calibration_do_not_mutate_scales(self):
        model = self._model(Qwen3_5MoeForCausalLM)
        malformed = Path(self.tmpdir.name) / "malformed.json"
        malformed.write_text("{")
        parallel = SimpleNamespace(tp_size=2, tp_rank=0)
        for path in (str(Path(self.tmpdir.name) / "missing.json"), str(malformed)):
            with (
                self.subTest(path=path),
                patch("sglang.srt.models.qwen3_5.get_parallel", return_value=parallel),
                self.assertLogs(
                    "sglang.srt.model_loader.weight_utils", level="WARNING"
                ),
            ):
                model.load_kv_cache_scales(path)
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
                wrapper.load_kv_cache_scales("scales.json")
                wrapper.model.load_kv_cache_scales.assert_called_once_with(
                    "scales.json"
                )


if __name__ == "__main__":
    unittest.main()
