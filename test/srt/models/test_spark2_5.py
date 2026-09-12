import unittest
from unittest.mock import patch

import torch
from torch import nn
from transformers.activations import ACT2FN

from sglang.srt.models.spark2_5 import Spark2_5MLP


class TestSpark2_5MLP(unittest.TestCase):
    def _make_mlp(self):
        with (
            patch(
                "sglang.srt.models.spark2_5.MergedColumnParallelLinear",
                return_value=nn.Identity(),
            ),
            patch(
                "sglang.srt.models.spark2_5.RowParallelLinear",
                return_value=nn.Identity(),
            ),
        ):
            return Spark2_5MLP(hidden_size=8, intermediate_size=16)

    def test_uses_reference_gelu(self):
        mlp = self._make_mlp()
        self.assertEqual(mlp.act_fn.approximate, "none")

    @unittest.skipUnless(torch.cuda.is_available(), "GPU is not available")
    def test_gpu_activation_matches_reference_gelu(self):
        mlp = self._make_mlp().cuda()
        torch.manual_seed(0)
        boundary = torch.tensor(
            [
                -float("inf"),
                -5.0,
                -3.0,
                -1.0,
                -0.0,
                0.0,
                1.0,
                3.0,
                5.0,
                float("inf"),
            ],
            device="cuda",
            dtype=torch.float32,
        )
        random = torch.randn(4096, device="cuda", dtype=torch.float32)
        gate = torch.cat((boundary, random))
        multiplier = torch.linspace(
            -2.0, 2.0, gate.numel(), device="cuda", dtype=torch.float32
        )
        fused_input = torch.cat((gate, multiplier))

        actual = mlp.act_fn(fused_input)
        reference = ACT2FN["gelu"](gate) * multiplier

        torch.testing.assert_close(
            actual, reference, rtol=1e-6, atol=1e-6, equal_nan=True
        )


if __name__ == "__main__":
    unittest.main()
