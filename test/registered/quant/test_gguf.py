import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from huggingface_hub import hf_hub_download

import sglang as sgl
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.quantization.gguf import GGUFConfig, GGUFMoEMethod
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=142, stage="base-b", runner_config="1-gpu-small")


class TestGGUFMoEMethod(unittest.TestCase):
    def setUp(self):
        self.method = GGUFMoEMethod(GGUFConfig())
        self.layer = SimpleNamespace(
            w13_qweight=torch.ones((1, 2, 2)),
            w2_qweight=torch.ones((1, 2, 2)),
            w13_qweight_type=SimpleNamespace(weight_type=2),
            w2_qweight_type=SimpleNamespace(weight_type=8),
        )
        self.dispatch_output = StandardDispatchOutput(
            hidden_states=torch.tensor([[1.0, 2.0]]),
            hidden_states_scale=None,
            topk_output=StandardTopKOutput(
                topk_weights=torch.tensor([[0.75]]),
                topk_ids=torch.tensor([[0]]),
                router_logits=None,
            ),
        )

    def test_apply_invokes_gguf_moe_kernel(self):
        self.method.create_moe_runner(None, SimpleNamespace(activation="silu"))
        expected = torch.tensor([[3.0, 4.0]])

        with patch(
            "sglang.srt.layers.quantization.gguf.fused_moe_gguf",
            return_value=expected,
        ) as fused_moe:
            result = self.method.apply(self.layer, self.dispatch_output)

        self.assertIs(result.hidden_states, expected)
        fused_moe.assert_called_once_with(
            x=self.dispatch_output.hidden_states,
            w1=self.layer.w13_qweight,
            w2=self.layer.w2_qweight,
            topk_weights=self.dispatch_output.topk_output.topk_weights,
            topk_ids=self.dispatch_output.topk_output.topk_ids,
            qweight_type=2,
            qweight_type2=8,
            activation="silu",
        )

    def test_apply_rejects_unsupported_activation_before_kernel(self):
        self.method.create_moe_runner(None, SimpleNamespace(activation="gelu"))

        with patch(
            "sglang.srt.layers.quantization.gguf.fused_moe_gguf"
        ) as fused_moe, self.assertRaisesRegex(
            AssertionError, "Only SiLU activation is supported"
        ):
            self.method.apply(self.layer, self.dispatch_output)

        fused_moe.assert_not_called()


class TestGGUF(CustomTestCase):
    def test_models(self):
        prompt = "Today is a sunny day and I like"
        sampling_params = {"temperature": 0, "max_new_tokens": 8}

        model_path = hf_hub_download(
            "Qwen/Qwen2-1.5B-Instruct-GGUF",
            filename="qwen2-1_5b-instruct-q4_k_m.gguf",
        )

        engine = sgl.Engine(
            model_path=model_path, random_seed=42, cuda_graph_max_bs_decode=2
        )
        outputs = engine.generate(prompt, sampling_params)["text"]
        engine.shutdown()

        self.assertEqual(outputs, " it. I have a lot of work")


if __name__ == "__main__":
    unittest.main()
