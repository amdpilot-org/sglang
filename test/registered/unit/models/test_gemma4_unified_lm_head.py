# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from transformers import Gemma4UnifiedConfig, Gemma4UnifiedTextConfig

from sglang.srt.models import gemma4_unified
from sglang.srt.models.gemma4_unified import (
    Gemma4UnifiedForConditionalGeneration,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class _FakeTextModel(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.start_layer = 0
        self.end_layer = config.num_hidden_layers

    def get_per_layer_inputs(self, input_ids):
        return None

    def tie_weights(self):
        return None


class _FakeLMHead(nn.Linear):
    def __init__(self, vocab_size, hidden_size, *args, **kwargs):
        super().__init__(hidden_size, vocab_size, bias=False)


class _FakeLogitsProcessor(nn.Module):
    def __init__(self, config):
        super().__init__()

    def forward(self, input_ids, hidden_states, head, forward_batch, aux):
        return hidden_states @ head.weight.T


class TestGemma4UnifiedLMHead(CustomTestCase):
    def _config(self, tied):
        text = Gemma4UnifiedTextConfig(
            vocab_size=17,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            global_head_dim=4,
            layer_types=["sliding_attention"],
            tie_word_embeddings=tied,
        )
        config = Gemma4UnifiedConfig(text_config=text)
        config.vision_config = None
        config.audio_config = None
        config.image_token_id = 14
        config.video_token_id = 15
        config.audio_token_id = 16
        config.eoi_token_id = None
        config.eoa_token_index = None
        return config

    def _build(self, tied, *, cpu=False, amx=False):
        pp_group = SimpleNamespace(world_size=1, is_first_rank=True, is_last_rank=True)
        with (
            patch.object(gemma4_unified, "get_pp_group", return_value=pp_group),
            patch.object(gemma4_unified, "Gemma4TextModel", _FakeTextModel),
            patch.object(gemma4_unified, "ParallelLMHead", _FakeLMHead),
            patch.object(gemma4_unified, "LogitsProcessor", _FakeLogitsProcessor),
            patch.object(gemma4_unified, "_is_cpu", cpu, create=True),
            patch.object(
                gemma4_unified, "_is_cpu_amx_available", amx, create=True
            ),
        ):
            return Gemma4UnifiedForConditionalGeneration(self._config(tied))

    def test_constructor_records_tied_head_and_first_forward_uses_embedding(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = self._build(tied=True).to(device)
        self.assertTrue(model.lm_head_is_tied)
        self.assertIs(model.lm_head, model.language_model.embed_tokens)

        hidden = torch.arange(16, dtype=torch.float32, device=device).reshape(2, 8)
        input_ids = torch.tensor([3, 4], device=device)
        positions = torch.tensor([0, 1], device=device)
        batch = SimpleNamespace(forward_mode=None, contains_image_inputs=lambda: False)
        with patch(
            "sglang.srt.models.gemma4_mm.general_mm_embed_routine",
            return_value=hidden,
        ):
            actual = model(input_ids, positions, batch)

        expected = hidden @ model.language_model.embed_tokens.weight.T
        torch.testing.assert_close(actual, expected)

    def test_constructor_records_untied_head_and_first_forward_uses_lm_head(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = self._build(tied=False).to(device)
        self.assertFalse(model.lm_head_is_tied)
        self.assertIsNot(model.lm_head, model.language_model.embed_tokens)

        hidden = torch.arange(16, dtype=torch.float32, device=device).reshape(2, 8)
        input_ids = torch.tensor([3, 4], device=device)
        positions = torch.tensor([0, 1], device=device)
        batch = SimpleNamespace(forward_mode=None, contains_image_inputs=lambda: False)
        with patch(
            "sglang.srt.models.gemma4_mm.general_mm_embed_routine",
            return_value=hidden,
        ):
            actual = model(input_ids, positions, batch)

        expected = hidden @ model.lm_head.weight.T
        torch.testing.assert_close(actual, expected)

    def test_cpu_amx_materializes_head_for_tied_checkpoint_loading(self):
        model = self._build(tied=True, cpu=True, amx=True)
        self.assertFalse(model.lm_head_is_tied)
        self.assertIsInstance(model.lm_head, _FakeLMHead)

        expected = torch.arange(136, dtype=torch.float32).reshape(17, 8)
        loaded = model.load_weights(
            [("model.language_model.embed_tokens.weight", expected)]
        )

        self.assertIn("language_model.embed_tokens.weight", loaded)
        self.assertIn("lm_head.weight", loaded)
        torch.testing.assert_close(model.language_model.embed_tokens.weight, expected)
        torch.testing.assert_close(model.lm_head.weight, expected)

    def test_untied_checkpoint_loads_embedding_and_head_independently(self):
        model = self._build(tied=False)
        embedding = torch.arange(136, dtype=torch.float32).reshape(17, 8)
        head = embedding.flip(0)

        loaded = model.load_weights(
            [
                ("model.language_model.embed_tokens.weight", embedding),
                ("model.lm_head.weight", head),
            ]
        )

        self.assertIn("language_model.embed_tokens.weight", loaded)
        self.assertIn("lm_head.weight", loaded)
        torch.testing.assert_close(model.language_model.embed_tokens.weight, embedding)
        torch.testing.assert_close(model.lm_head.weight, head)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA or ROCm")
    def test_tied_head_forward_captures_and_replays_on_gpu(self):
        model = self._build(tied=True).cuda()
        hidden = torch.arange(16, dtype=torch.float32, device="cuda").reshape(2, 8)
        input_ids = torch.tensor([3, 4], device="cuda")
        positions = torch.tensor([0, 1], device="cuda")
        batch = SimpleNamespace(forward_mode=None, contains_image_inputs=lambda: False)

        graph = torch.cuda.CUDAGraph()
        with patch(
            "sglang.srt.models.gemma4_mm.general_mm_embed_routine",
            return_value=hidden,
        ):
            with torch.cuda.graph(graph):
                actual = model(input_ids, positions, batch)
        graph.replay()

        expected = hidden @ model.language_model.embed_tokens.weight.T
        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
