"""Regression coverage for Qwen4-Exp DFlash hidden-state capture."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.models.qwen3_vl import Qwen3VLForConditionalGeneration
from sglang.srt.models.qwen4_exp import (
    Qwen4ExpForConditionalGeneration,
    Qwen4ExpModel,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _ForwardMode:
    def __init__(self, idle: bool):
        self._idle = idle

    def is_idle(self):
        return self._idle


class _DeterministicQwen4Layer(nn.Module):
    """Small stand-in for the layer body, preserving Qwen4's HC layout."""

    def __init__(self, hc_count: int, hidden_size: int, offset: float):
        super().__init__()
        self.hc_count = hc_count
        self.hidden_size = hidden_size
        self.offset = offset
        self.attn_hyper_connection = _LearnedMix(
            hc_count, hidden_size, scale=offset
        )

    def forward(self, hidden_states, residual, **kwargs):
        if hidden_states.shape[-1] == self.hidden_size:
            hidden_states = hidden_states.repeat(1, self.hc_count)
        branch_offsets = torch.arange(
            self.hc_count, dtype=hidden_states.dtype, device=hidden_states.device
        ).repeat_interleave(self.hidden_size)
        return hidden_states + branch_offsets + self.offset, None


class _LearnedMix(nn.Module):
    def __init__(self, hc_count: int, hidden_size: int, scale: float):
        super().__init__()
        self.hc_count = hc_count
        self.hidden_size = hidden_size
        self.register_buffer(
            "weights",
            torch.arange(1, hc_count + 1, dtype=torch.float32) * scale,
        )

    def mix(self, hidden_states):
        branches = hidden_states.unflatten(
            -1, (self.hc_count, self.hidden_size)
        )
        mixed = (branches * self.weights.view(1, -1, 1)).mean(-2)
        return mixed, (hidden_states, hidden_states)


class _FinalMixer(nn.Module):
    def __init__(self, hc_count: int):
        super().__init__()
        self.hc_count = hc_count

    def mix(self, hidden_states):
        return hidden_states.unflatten(-1, (self.hc_count, -1)).mean(-2), hidden_states


def _make_model():
    model = Qwen4ExpModel.__new__(Qwen4ExpModel)
    nn.Module.__init__(model)
    model.hc_count = 2
    model.hidden_size = 3
    model.has_ple = False
    model.ple_ngram_size = None
    model.ple_ngram_eos_token_id = None
    model._start_layer = 0
    model._end_layer = 3
    model.dflash_capture = False
    model.layers = nn.ModuleList(
        [_DeterministicQwen4Layer(2, 3, float(i + 1)) for i in range(3)]
    )
    model.hyper_connection_mixer = _FinalMixer(2)
    return model


class TestQwen4ExpDflashCapture(CustomTestCase):
    def _forward(self, model, *, idle=False):
        return model.forward(
            input_ids=torch.zeros(2, dtype=torch.long),
            positions=torch.arange(2),
            forward_batch=SimpleNamespace(forward_mode=_ForwardMode(idle)),
            inputs_embeds=torch.arange(6, dtype=torch.float32).reshape(2, 3),
        )

    def test_selected_taps_are_contracted_in_layer_order(self):
        model = _make_model()
        model.set_dflash_layers_to_capture([1, 2])

        final_hidden, hc_hidden, captures = self._forward(model)

        # Independent layer-by-layer reference. A flag on layer k captures its
        # input, which is the completed output of layer k - 1.
        state = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        states = []
        for layer_id, layer in enumerate(model.layers):
            if state.shape[-1] == model.hidden_size:
                state = state.repeat(1, model.hc_count)
            offsets = torch.arange(model.hc_count).repeat_interleave(model.hidden_size)
            state = state + offsets + layer.offset
            states.append(state.clone())

        # Captures occur before layers 1 and 2. Compute their learned mixes
        # independently rather than calling the implementation under test.
        expected = []
        for layer_id, state_before_layer in zip((1, 2), states[:2]):
            branches = state_before_layer.unflatten(
                -1, (model.hc_count, model.hidden_size)
            )
            weights = torch.arange(1, model.hc_count + 1, dtype=torch.float32)
            expected.append(
                (branches * (weights * model.layers[layer_id].offset).view(1, -1, 1))
                .mean(-2)
            )

        self.assertEqual(len(captures), 2)
        for actual, reference in zip(captures, expected):
            torch.testing.assert_close(actual, reference)
            self.assertEqual(tuple(actual.shape), (2, model.hidden_size))
        uniform = states[0].unflatten(-1, (2, 3)).mean(-2)
        self.assertGreater((captures[0] - uniform).abs().max().item(), 1e-3)
        torch.testing.assert_close(hc_hidden, state)
        torch.testing.assert_close(final_hidden, state.unflatten(-1, (2, 3)).mean(-2))

    def test_ordinary_and_idle_return_shapes_are_unchanged(self):
        model = _make_model()

        normal = self._forward(model)
        self.assertIsInstance(normal, tuple)
        self.assertEqual(len(normal), 2)

        idle = self._forward(model, idle=True)
        self.assertIsInstance(idle, torch.Tensor)

    def test_aux_output_is_not_overwritten_by_hc_stream(self):
        wrapper = Qwen4ExpForConditionalGeneration.__new__(
            Qwen4ExpForConditionalGeneration
        )
        nn.Module.__init__(wrapper)
        wrapper.model = nn.Module()
        wrapper.model.last_hc_hidden_states = torch.full((1, 6), 99.0)
        aux = torch.arange(6, dtype=torch.float32).reshape(1, 6)

        wrapper.capture_aux_hidden_states = True
        output = LogitsProcessorOutput(next_token_logits=None, hidden_states=aux)
        with patch.object(Qwen3VLForConditionalGeneration, "forward", return_value=output):
            actual = wrapper.forward()
        self.assertIs(actual.hidden_states, aux)

        wrapper.capture_aux_hidden_states = False
        output = LogitsProcessorOutput(next_token_logits=None, hidden_states=None)
        with patch.object(Qwen3VLForConditionalGeneration, "forward", return_value=output):
            actual = wrapper.forward()
        self.assertIs(actual.hidden_states, wrapper.model.last_hc_hidden_states)


if __name__ == "__main__":
    unittest.main()
