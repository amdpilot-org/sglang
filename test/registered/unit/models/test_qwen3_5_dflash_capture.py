import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.moe.qwen35_flashinfer_fusion import (
    Qwen35MoeFinalizeHandoff,
)
from sglang.srt.models.qwen2 import Qwen2Model
from sglang.srt.models.qwen3 import Qwen3Model
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

    def test_backbone_reconfiguration_clears_old_layer_flags(self):
        model = Qwen3_5ForCausalLM.__new__(Qwen3_5ForCausalLM)
        torch.nn.Module.__init__(model)
        model.layers = torch.nn.ModuleList([torch.nn.Identity() for _ in range(3)])

        model.set_dflash_layers_to_capture([1])
        model.set_dflash_layers_to_capture([2])

        self.assertFalse(hasattr(model.layers[1], "_is_layer_to_capture"))
        self.assertTrue(model.layers[2]._is_layer_to_capture)

    def test_terminal_capture_returns_pre_norm_residual_stream(self):
        model = Qwen3_5ForCausalLM.__new__(Qwen3_5ForCausalLM)
        model._capture_after_last_layer = True
        hidden_states = torch.tensor([[1.0, 2.0]])
        final_decoder_output = torch.tensor([[11.0, 22.0]])
        captured = []

        model._capture_final_decoder_output(final_decoder_output, captured)

        torch.testing.assert_close(captured[0], torch.tensor([[11.0, 22.0]]))

    def test_terminal_capture_without_residual_returns_hidden_states(self):
        model = Qwen3_5ForCausalLM.__new__(Qwen3_5ForCausalLM)
        model._capture_after_last_layer = True
        hidden_states = torch.tensor([[1.0, 2.0]])
        captured = []

        model._capture_final_decoder_output(hidden_states, captured)

        self.assertIs(captured[0], hidden_states)

    def test_terminal_capture_accepts_deferred_finalize_output(self):
        class FakeFusion:
            def finalize(self, handoff, residual, gamma):
                self.handoff = handoff
                return torch.tensor([[1.0, 2.0]]), torch.tensor([[11.0, 22.0]])

        model = Qwen3_5ForCausalLM.__new__(Qwen3_5ForCausalLM)
        torch.nn.Module.__init__(model)
        model.pp_group = SimpleNamespace(is_first_rank=False, is_last_rank=True)
        model.layers = torch.nn.ModuleList()
        model._start_layer = 0
        model._end_layer = 0
        model._capture_after_last_layer = True
        model.flashinfer_mnnvl_cutedsl_fusion = FakeFusion()
        model.norm = SimpleNamespace(gemma_weight=torch.tensor([1.0, 1.0]))
        handoff = Qwen35MoeFinalizeHandoff(
            routed_output=torch.empty(0),
            expert_weights=torch.empty(0),
            permuted_indices=torch.empty(0, dtype=torch.int32),
            gated_shared_output=torch.empty(0),
            m=1,
        )

        output, captured = model(
            input_ids=torch.tensor([0]),
            positions=torch.tensor([0]),
            forward_batch=SimpleNamespace(),
            pp_proxy_tensors={
                "hidden_states": handoff,
                "residual": torch.tensor([[10.0, 20.0]]),
            },
        )

        torch.testing.assert_close(output, torch.tensor([[1.0, 2.0]]))
        torch.testing.assert_close(captured[0], torch.tensor([[11.0, 22.0]]))
        self.assertIs(model.flashinfer_mnnvl_cutedsl_fusion.handoff, handoff)

    def test_qwen2_and_qwen3_capture_terminal_position(self):
        class AddOneLayer(torch.nn.Module):
            def forward(self, positions, hidden_states, forward_batch, residual):
                return hidden_states + 1, residual

        for model_cls in (Qwen2Model, Qwen3Model):
            with self.subTest(model=model_cls.__name__):
                model = model_cls.__new__(model_cls)
                torch.nn.Module.__init__(model)
                model.pp_group = SimpleNamespace(is_first_rank=True, is_last_rank=True)
                model.layers = torch.nn.ModuleList([AddOneLayer() for _ in range(3)])
                model.start_layer = 0
                model.end_layer = 3
                model.layers_to_capture = [3]
                model.norm = torch.nn.Identity()

                output, captured = model(
                    input_ids=torch.tensor([0]),
                    positions=torch.tensor([0]),
                    forward_batch=SimpleNamespace(),
                    input_embeds=torch.tensor([[0.0]]),
                )

                torch.testing.assert_close(output, torch.tensor([[3.0]]))
                torch.testing.assert_close(captured[0], torch.tensor([[3.0]]))


if __name__ == "__main__":
    unittest.main()
