"""Online FP8 weight quantization versus an equivalent prepared checkpoint."""

import unittest

import torch

from sglang.kernels.ops.quantization.fp8_kernel import is_fp8_fnuz, scaled_fp8_quant
from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-large")


class TestFp8OnlinePreparedLinear(unittest.TestCase):
    """Validate per-tensor FP8 online and prepared linear equivalence.

    The prepared checkpoint uses the online path's exact FP8 codes. On FNUZ
    devices, the checkpoint stores e4m3fn codes with half the runtime scale;
    ``process_weights_after_loading`` normalizes that representation back to
    e4m3fnuz and the identical runtime scale.
    """

    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")

    @staticmethod
    def _make_layer(config, weight, checkpoint_weight=None, checkpoint_scale=None):
        layer = torch.nn.Module()
        method = Fp8LinearMethod(config)
        method.create_weights(
            layer,
            input_size_per_partition=weight.shape[1],
            output_partition_sizes=[weight.shape[0]],
            input_size=weight.shape[1],
            output_size=weight.shape[0],
            params_dtype=torch.bfloat16,
        )
        layer.to(weight.device)
        layer.weight.data.copy_(
            checkpoint_weight if checkpoint_weight is not None else weight
        )
        if checkpoint_scale is not None:
            layer.weight_scale.data.copy_(checkpoint_scale)
        method.process_weights_after_loading(layer)
        return method, layer

    def test_online_quantization_matches_prepared_checkpoint(self):
        torch.manual_seed(15194)
        device = torch.device("cuda")
        num_tokens, output_size, input_size = 128, 128, 256
        weight = torch.randn(
            output_size, input_size, device=device, dtype=torch.bfloat16
        ).div_(10)
        activation = torch.randn(
            num_tokens, input_size, device=device, dtype=torch.bfloat16
        ).div_(10)

        online_method, online_layer = self._make_layer(Fp8Config(), weight)
        online_weight = online_layer.weight.data
        online_scale = online_layer.weight_scale.data

        checkpoint_weight = (
            online_weight.t()
            .contiguous()
            .view(torch.int8)
            .view(torch.float8_e4m3fn)
        )
        checkpoint_weight.view(torch.int8)[
            checkpoint_weight.view(torch.int8) == -128
        ] = 0
        checkpoint_scale = online_scale.reshape(1) * (0.5 if is_fp8_fnuz() else 1.0)
        serialized_config = Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
        )
        prepared_method, prepared_layer = self._make_layer(
            serialized_config,
            weight,
            checkpoint_weight=checkpoint_weight,
            checkpoint_scale=checkpoint_scale,
        )
        prepared_weight = prepared_layer.weight.data
        prepared_scale = prepared_layer.weight_scale.data

        self.assertTrue(
            torch.equal(
                online_weight.view(torch.int8), prepared_weight.view(torch.int8)
            )
        )
        self.assertTrue(torch.equal(online_scale, prepared_scale))

        online_output = online_method.apply(online_layer, activation)
        prepared_output = prepared_method.apply(prepared_layer, activation)
        self.assertTrue(torch.equal(online_output, prepared_output))

        quantized_activation, activation_scale = scaled_fp8_quant(activation)
        reference = (
            quantized_activation.float() * activation_scale
        ) @ (online_weight.float() * online_scale)
        reference = reference.to(online_output.dtype)
        torch.testing.assert_close(
            online_output,
            reference,
            rtol=2e-2,
            atol=1e-2,
        )


if __name__ == "__main__":
    unittest.main()
