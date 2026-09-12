import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.models.qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
    Qwen3_5MoeForConditionalGeneration,
    _get_decoder_layer_id,
)
from sglang.srt.models.qwen3_5_mtp import Qwen3_5ForCausalLMMTP
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestQwen3_5PipelineParallel(CustomTestCase):
    def test_decoder_layer_id_excludes_encoder_namespaces(self):
        self.assertEqual(_get_decoder_layer_id("model.layers.7.mlp.weight"), 7)
        self.assertIsNone(_get_decoder_layer_id("model.visual.layers.7.weight"))
        self.assertIsNone(_get_decoder_layer_id("audio_tower.layers.7.weight"))
        self.assertIsNone(_get_decoder_layer_id("layers.7.weight"))

    @staticmethod
    def _make_vl_weight_loader_stub():
        model = Qwen3_5ForConditionalGeneration.__new__(Qwen3_5ForConditionalGeneration)
        torch.nn.Module.__init__(model)
        model.model = torch.nn.Module()
        model.model.layers = torch.nn.ModuleList(
            [torch.nn.Linear(1, 1, bias=False) for _ in range(4)]
        )
        model.model.start_layer = 1
        model.model.end_layer = 3
        model.visual = torch.nn.Module()
        model.visual.layers = torch.nn.ModuleList(
            [torch.nn.Linear(1, 1, bias=False) for _ in range(4)]
        )
        model.config = SimpleNamespace(tie_word_embeddings=False)
        model.pp_group = SimpleNamespace(is_last_rank=False)
        with torch.no_grad():
            for param in model.parameters():
                param.fill_(torch.nan)
        return model

    @staticmethod
    def _make_mtp_weight_loader_stub():
        model = Qwen3_5ForCausalLMMTP.__new__(Qwen3_5ForCausalLMMTP)
        torch.nn.Module.__init__(model)
        model.model = torch.nn.Module()
        model.model.embed_tokens = torch.nn.Embedding(4, 3)
        model.config = SimpleNamespace(num_experts=None)
        model.quant_config = None
        with torch.no_grad():
            model.model.embed_tokens.weight.fill_(torch.nan)
        return model

    @staticmethod
    def _get_num_fused_shared_experts(layers, start_layer, end_layer):
        model = SimpleNamespace(
            model=SimpleNamespace(
                layers=layers,
                start_layer=start_layer,
                end_layer=end_layer,
            )
        )
        return Qwen3_5MoeForConditionalGeneration._get_num_fused_shared_experts(model)

    def test_get_num_fused_shared_experts_returns_zero_without_layers(self):
        model = SimpleNamespace(model=SimpleNamespace())

        num_fused_shared_experts = (
            Qwen3_5MoeForConditionalGeneration._get_num_fused_shared_experts(model)
        )

        self.assertEqual(num_fused_shared_experts, 0)

    def test_get_num_fused_shared_experts_uses_local_pp_layers(self):
        layers = [
            PPMissingLayer(),
            PPMissingLayer(),
            SimpleNamespace(
                mlp=SimpleNamespace(num_fused_shared_experts=1),
            ),
            SimpleNamespace(
                mlp=SimpleNamespace(num_fused_shared_experts=1),
            ),
        ]

        num_fused_shared_experts = self._get_num_fused_shared_experts(
            layers,
            start_layer=2,
            end_layer=4,
        )

        self.assertEqual(num_fused_shared_experts, 1)

    def test_get_num_fused_shared_experts_returns_zero_without_local_fusion(self):
        layers = [
            PPMissingLayer(),
            SimpleNamespace(mlp=SimpleNamespace()),
        ]

        num_fused_shared_experts = self._get_num_fused_shared_experts(
            layers,
            start_layer=1,
            end_layer=2,
        )

        self.assertEqual(num_fused_shared_experts, 0)

    def test_mtp_loads_vl_target_embedding_for_last_pp_stage(self):
        model = self._make_mtp_weight_loader_stub()
        expected = torch.arange(12, dtype=torch.float32).reshape(4, 3)

        loaded = model.load_weights(
            [("model.language_model.embed_tokens.weight", expected)]
        )

        self.assertEqual(loaded, {"model.embed_tokens.weight"})
        torch.testing.assert_close(model.model.embed_tokens.weight, expected)

    def test_mtp_loads_text_target_embedding_for_last_pp_stage(self):
        model = self._make_mtp_weight_loader_stub()
        expected = torch.arange(12, dtype=torch.float32).reshape(4, 3)

        loaded = model.load_weights([("model.embed_tokens.weight", expected)])

        self.assertEqual(loaded, {"model.embed_tokens.weight"})
        torch.testing.assert_close(model.model.embed_tokens.weight, expected)

    def test_vl_load_weights_filters_only_decoder_layer_range(self):
        model = self._make_vl_weight_loader_stub()
        weights = [
            ("model.layers.0.weight", torch.tensor([[10.0]])),
            ("model.layers.1.weight", torch.tensor([[11.0]])),
            ("model.layers.2.weight", torch.tensor([[12.0]])),
            ("model.layers.3.weight", torch.tensor([[13.0]])),
            ("model.visual.layers.3.weight", torch.tensor([[23.0]])),
        ]

        loaded = model.load_weights(weights)

        self.assertEqual(
            loaded,
            {
                "model.layers.1.weight",
                "model.layers.2.weight",
                "visual.layers.3.weight",
            },
        )
        self.assertTrue(torch.isnan(model.model.layers[0].weight).all())
        torch.testing.assert_close(model.model.layers[1].weight, torch.tensor([[11.0]]))
        torch.testing.assert_close(model.model.layers[2].weight, torch.tensor([[12.0]]))
        self.assertTrue(torch.isnan(model.model.layers[3].weight).all())
        torch.testing.assert_close(
            model.visual.layers[3].weight, torch.tensor([[23.0]])
        )


if __name__ == "__main__":
    unittest.main()
