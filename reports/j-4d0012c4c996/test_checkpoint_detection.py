import json
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups.model_overrides.deepseek_v4 import _deepseek_v4_overrides
from sglang.srt.configs.deepseek_v4 import try_detect_fp4_experts
from sglang.srt.environ import envs


class TestDeepSeekV4CheckpointDetection(unittest.TestCase):
    def make_checkpoint(self, dtype=None):
        tmp = tempfile.TemporaryDirectory()
        root = Path(tmp.name)
        key = "layers.0.ffn.experts.0.w1.weight"
        shard = "model-00001-of-00001.safetensors"
        (root / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {key: shard}})
        )
        header = {} if dtype is None else {key: {"dtype": dtype, "shape": [2, 2], "data_offsets": [0, 4]}}
        encoded = json.dumps(header).encode()
        (root / shard).write_bytes(struct.pack("<Q", len(encoded)) + encoded)
        return tmp, root

    def test_packed_fp4_storage_is_detected(self):
        tmp, root = self.make_checkpoint("I8")
        with tmp:
            self.assertIs(try_detect_fp4_experts(str(root)), True)

    def test_true_fp8_storage_is_not_misclassified(self):
        tmp, root = self.make_checkpoint("F8_E4M3")
        with tmp:
            self.assertIs(try_detect_fp4_experts(str(root)), False)

    def test_missing_header_entry_is_inconclusive(self):
        tmp, root = self.make_checkpoint()
        with tmp:
            self.assertIsNone(try_detect_fp4_experts(str(root)))

    def test_remote_slug_is_inconclusive_without_cache(self):
        self.assertIsNone(try_detect_fp4_experts("org/not-a-local-checkpoint"))

    def test_hopper_auto_route_selects_mxfp4(self):
        args = SimpleNamespace(
            device="cuda",
            swa_full_tokens_ratio=None,
            moe_a2a_backend="none",
            moe_runner_backend="auto",
            _model_config=SimpleNamespace(is_fp4_experts=True, nvfp4_moe_meta=None),
        )
        hf = SimpleNamespace(architectures=["DeepseekV4ForCausalLM"])
        hopper = SimpleNamespace(
            is_hip=False, is_sm90=True, is_sm100=False, is_sm120=False
        )
        with (
            envs.SGLANG_DSV4_FP4_DEQUANT.override(False),
            patch(
                "sglang.srt.arg_groups.model_overrides.deepseek_v4.get_platform",
                return_value=hopper,
            ),
        ):
            result = _deepseek_v4_overrides(args, hf)
        self.assertEqual(result["moe_runner_backend"], "flashinfer_mxfp4")

    def test_true_fp8_hopper_keeps_generic_route(self):
        args = SimpleNamespace(
            device="cuda",
            swa_full_tokens_ratio=None,
            moe_a2a_backend="none",
            moe_runner_backend="auto",
            _model_config=SimpleNamespace(is_fp4_experts=False, nvfp4_moe_meta=None),
        )
        hf = SimpleNamespace(architectures=["DeepseekV4ForCausalLM"])
        hopper = SimpleNamespace(
            is_hip=False, is_sm90=True, is_sm100=False, is_sm120=False
        )
        with patch(
            "sglang.srt.arg_groups.model_overrides.deepseek_v4.get_platform",
            return_value=hopper,
        ):
            result = _deepseek_v4_overrides(args, hf)
        self.assertNotIn("moe_runner_backend", result)


if __name__ == "__main__":
    unittest.main()
