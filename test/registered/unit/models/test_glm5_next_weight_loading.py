import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.model_loader.runai_utils import RUNAI_STREAMER_TENSOR_ATTR
from sglang.srt.models.glm5_next import Glm5NextForConditionalGeneration
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _FakeParam:
    def __init__(self):
        self.loaded = None

    def weight_loader(self, _param, loaded_weight):
        self.loaded = loaded_weight


class TestGlm5NextWeightLoading(unittest.TestCase):
    def _load_fused(self, weights):
        fused_param = _FakeParam()
        model = SimpleNamespace(
            config=SimpleNamespace(n_routed_experts=0, num_hidden_layers=1),
            num_fused_shared_experts=0,
            quant_config=None,
            fuse_qkv_a_proj=True,
            named_parameters=lambda: iter(
                [
                    (
                        "model.layers.0.self_attn.fused_qkv_a_proj_with_mqa.weight",
                        fused_param,
                    )
                ]
            ),
        )
        with patch(
            "sglang.srt.models.glm5_next.DeepseekV2WeightLoaderMixin.post_load_weights"
        ) as post_load:
            Glm5NextForConditionalGeneration.load_weights(model, weights)
            post_load.assert_called_once()
        return fused_param.loaded

    def test_runai_streamed_fused_qkv_a_proj_owns_reused_buffer(self):
        staging_buffer = torch.tensor([1, 2], dtype=torch.float32)

        def streamed_weights():
            q_view = staging_buffer[:]
            setattr(q_view, RUNAI_STREAMER_TENSOR_ATTR, True)
            yield "model.layers.0.self_attn.q_a_proj.weight", q_view

            staging_buffer.copy_(torch.tensor([3, 4], dtype=torch.float32))
            kv_view = staging_buffer[:]
            setattr(kv_view, RUNAI_STREAMER_TENSOR_ATTR, True)
            yield "model.layers.0.self_attn.kv_a_proj_with_mqa.weight", kv_view

        fused = self._load_fused(streamed_weights())

        self.assertEqual(fused.tolist(), [1, 2, 3, 4])

    def test_runai_streamed_fused_qkv_a_proj_owns_reused_buffer_kv_first(self):
        staging_buffer = torch.tensor([3, 4], dtype=torch.float32)

        def streamed_weights():
            kv_view = staging_buffer[:]
            setattr(kv_view, RUNAI_STREAMER_TENSOR_ATTR, True)
            yield "model.layers.0.self_attn.kv_a_proj_with_mqa.weight", kv_view

            staging_buffer.copy_(torch.tensor([1, 2], dtype=torch.float32))
            q_view = staging_buffer[:]
            setattr(q_view, RUNAI_STREAMER_TENSOR_ATTR, True)
            yield "model.layers.0.self_attn.q_a_proj.weight", q_view

        fused = self._load_fused(streamed_weights())

        self.assertEqual(fused.tolist(), [1, 2, 3, 4])

    def test_unmarked_fused_qkv_a_proj_preserves_values(self):
        weights = iter(
            [
                (
                    "model.layers.0.self_attn.q_a_proj.weight",
                    torch.tensor([1, 2], dtype=torch.float32),
                ),
                (
                    "model.layers.0.self_attn.kv_a_proj_with_mqa.weight",
                    torch.tensor([3, 4], dtype=torch.float32),
                ),
            ]
        )

        fused = self._load_fused(weights)

        self.assertEqual(fused.tolist(), [1, 2, 3, 4])


if __name__ == "__main__":
    unittest.main()
