import unittest
from types import SimpleNamespace

import torch

from sglang.srt.models.qwen3_5 import Qwen3_5ForCausalLM
from sglang.srt.models.qwen3_vl import Qwen3VLForConditionalGeneration
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestQwen3_5DFlashCapture(CustomTestCase):
    def test_vl_wrapper_shifts_last_hf_layer_to_terminal_position(self):
        captured_positions = []
        model = SimpleNamespace(
            pp_group=SimpleNamespace(is_last_rank=True),
            model=SimpleNamespace(
                set_dflash_layers_to_capture=captured_positions.extend
            ),
            capture_aux_hidden_states=False,
        )

        Qwen3VLForConditionalGeneration.set_dflash_layers_to_capture(model, [0, 2])

        self.assertEqual(captured_positions, [1, 3])
        self.assertTrue(model.capture_aux_hidden_states)

    def test_backbone_accepts_capture_position_after_last_layer(self):
        model = Qwen3_5ForCausalLM.__new__(Qwen3_5ForCausalLM)
        torch.nn.Module.__init__(model)
        model.layers = torch.nn.ModuleList([torch.nn.Identity() for _ in range(3)])

        model.set_dflash_layers_to_capture([1, 3])

        self.assertTrue(model.layers[1]._is_layer_to_capture)
        self.assertTrue(model._capture_after_last_layer)

    def test_backbone_rejects_capture_position_past_model_boundary(self):
        model = Qwen3_5ForCausalLM.__new__(Qwen3_5ForCausalLM)
        torch.nn.Module.__init__(model)
        model.layers = torch.nn.ModuleList([torch.nn.Identity() for _ in range(3)])

        with self.assertRaisesRegex(ValueError, r"in \[0, 3\].*got 4"):
            model.set_dflash_layers_to_capture([4])

    def test_terminal_capture_returns_pre_norm_residual_stream(self):
        model = Qwen3_5ForCausalLM.__new__(Qwen3_5ForCausalLM)
        model._capture_after_last_layer = True
        hidden_states = torch.tensor([[1.0, 2.0]])
        residual = torch.tensor([[10.0, 20.0]])
        captured = []

        model._capture_final_decoder_output(hidden_states, residual, captured)

        torch.testing.assert_close(captured[0], torch.tensor([[11.0, 22.0]]))

    def test_terminal_capture_without_residual_returns_hidden_states(self):
        model = Qwen3_5ForCausalLM.__new__(Qwen3_5ForCausalLM)
        model._capture_after_last_layer = True
        hidden_states = torch.tensor([[1.0, 2.0]])
        captured = []

        model._capture_final_decoder_output(hidden_states, None, captured)

        self.assertIs(captured[0], hidden_states)


if __name__ == "__main__":
    unittest.main()
