"""Tests for FP8 KV-cache scale loading diagnostics."""

import json
import logging
import tempfile
import unittest
from pathlib import Path

import torch

from sglang.kernels.ops.quantization.fp8_kernel import is_fp8_fnuz
from sglang.srt.layers.quantization.kv_cache import BaseKVCacheMethod
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.model_executor.model_runner_components.load_model_utils import (
    load_kv_cache_scales,
)
from sglang.srt.model_loader.weight_utils import kv_cache_scales_loader
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

LOGGER_NAME = "sglang.srt.model_executor.model_runner_components.load_model_utils"
FALLBACK_WARNING = "Defaulting to scaling factors of 1.0"


class _KVCacheQuantConfig:
    def get_quant_method(self, layer, prefix):
        return BaseKVCacheMethod(self)


def _attention(
    layer_id: int, scales: tuple[float, float] | None, *, quantized: bool = True
):
    layer = RadixAttention(
        num_heads=1,
        head_dim=8,
        scaling=1.0,
        num_kv_heads=1,
        layer_id=layer_id,
        quant_config=_KVCacheQuantConfig() if quantized else None,
    )
    if not quantized:
        return layer
    if scales is not None:
        layer.k_scale.data.fill_(scales[0])
        layer.v_scale.data.fill_(scales[1])
    layer.quant_method.process_weights_after_loading(layer)
    return layer


class _Model(torch.nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)
        self.external_scale_path = None

    def load_kv_cache_scales(self, path):
        self.external_scale_path = path
        for layer_id, scale in kv_cache_scales_loader(
            path,
            tp_rank=0,
            tp_size=1,
            num_hidden_layers=len(self.layers),
            model_type="llama",
        ):
            self.layers[layer_id].k_scale = scale
            self.layers[layer_id].v_scale = scale


class TestLoadKVCacheScales(CustomTestCase):
    def _load(self, model, *, quantization_param_path=None):
        with get_context().override_server_args(
            quantization_param_path=quantization_param_path
        ):
            load_kv_cache_scales(model=model, kv_cache_dtype="fp8_e4m3")

    def test_legacy_external_scales_are_loaded(self):
        model = _Model([_attention(0, None, quantized=False)])
        payload = {
            "model_type": "llama",
            "kv_cache": {
                "dtype": "float8_e4m3fn",
                "scaling_factor": {"0": {"0": 0.125}},
            },
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            scale_path = Path(temp_dir) / "scales.json"
            scale_path.write_text(json.dumps(payload))
            with self.assertLogs(LOGGER_NAME, level=logging.INFO) as logs:
                self._load(model, quantization_param_path=str(scale_path))
            self.assertEqual(model.external_scale_path, str(scale_path))
        expected = 0.25 if is_fp8_fnuz() else 0.125
        self.assertEqual(model.layers[0].k_scale_float, expected)
        self.assertEqual(model.layers[0].v_scale_float, expected)
        self.assertFalse(any(FALLBACK_WARNING in line for line in logs.output))

    def test_baked_per_layer_scales_do_not_warn(self):
        model = _Model([_attention(0, (0.028, 0.012)), _attention(1, (1.0, 1.0))])
        with self.assertLogs(LOGGER_NAME, level=logging.INFO) as logs:
            self._load(model)
        self.assertAlmostEqual(model.layers[0].k_scale_float, 0.028)
        self.assertAlmostEqual(model.layers[0].v_scale_float, 0.012)
        self.assertEqual(model.layers[1].k_scale_float, 1.0)
        self.assertEqual(model.layers[1].v_scale_float, 1.0)
        self.assertFalse(any(FALLBACK_WARNING in line for line in logs.output))

    def test_absent_scales_warn_and_use_defaults(self):
        model = _Model([_attention(0, None), _attention(1, None)])
        with self.assertLogs(LOGGER_NAME, level=logging.WARNING) as logs:
            self._load(model)
        self.assertTrue(any(FALLBACK_WARNING in line for line in logs.output))
        self.assertEqual(model.layers[0].k_scale_float, 1.0)
        self.assertEqual(model.layers[0].v_scale_float, 1.0)

    def test_partially_missing_scales_still_warn(self):
        model = _Model([_attention(0, (0.028, 0.012)), _attention(1, None)])
        with self.assertLogs(LOGGER_NAME, level=logging.WARNING) as logs:
            self._load(model)
        self.assertTrue(any("1 of 2" in line for line in logs.output))
        self.assertAlmostEqual(model.layers[0].k_scale_float, 0.028)
        self.assertEqual(model.layers[1].k_scale_float, 1.0)


if __name__ == "__main__":
    unittest.main()
