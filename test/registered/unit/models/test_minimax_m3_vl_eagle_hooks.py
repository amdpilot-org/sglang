import unittest
from types import SimpleNamespace

import torch

from sglang.srt.models.minimax_m3_vl import MiniMaxM3SparseForConditionalGeneration
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _fake_wrapper(last_rank: bool = True):
    embed_weight = torch.empty(2, 3)
    head_weight = torch.empty(3, 4)
    layers = [SimpleNamespace() for _ in range(6)]
    wrapper = SimpleNamespace(
        config=SimpleNamespace(
            text_config=SimpleNamespace(num_hidden_layers=len(layers))
        ),
        model=SimpleNamespace(
            embed_tokens=SimpleNamespace(weight=embed_weight),
            layers=layers,
        ),
        lm_head=SimpleNamespace(weight=head_weight),
        pp_group=SimpleNamespace(is_last_rank=last_rank),
        capture_aux_hidden_states=False,
    )
    return wrapper, embed_weight, head_weight


class TestMiniMaxM3VLEagleHooks(unittest.TestCase):
    def test_explicit_layer_capture_and_embed_head(self):
        wrapper, embed_weight, head_weight = _fake_wrapper()

        MiniMaxM3SparseForConditionalGeneration.set_eagle3_layers_to_capture(
            wrapper, [1, 3]
        )
        embed, head = MiniMaxM3SparseForConditionalGeneration.get_embed_and_head(
            wrapper
        )

        self.assertTrue(wrapper.capture_aux_hidden_states)
        self.assertEqual(wrapper.model.layers_to_capture, [2, 4])
        self.assertTrue(wrapper.model.layers[2]._is_layer_to_capture)
        self.assertTrue(wrapper.model.layers[4]._is_layer_to_capture)
        self.assertIs(embed, embed_weight)
        self.assertIs(head, head_weight)

    def test_non_last_rank_does_not_capture(self):
        wrapper, _, _ = _fake_wrapper(last_rank=False)

        MiniMaxM3SparseForConditionalGeneration.set_eagle3_layers_to_capture(
            wrapper, [1, 3]
        )

        self.assertFalse(wrapper.capture_aux_hidden_states)
        self.assertFalse(hasattr(wrapper.model, "layers_to_capture"))


if __name__ == "__main__":
    unittest.main()
