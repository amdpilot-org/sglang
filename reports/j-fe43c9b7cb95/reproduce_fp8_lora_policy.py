"""GPU evidence for the diffusion online-FP8 LoRA merge policy."""

from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.multimodal_gen.runtime.layers.linear import ReplicatedLinear
from sglang.multimodal_gen.runtime.layers.lora.linear import BaseLayerWithLoRA
from sglang.multimodal_gen.runtime.layers.quantization.fp8 import Fp8Config
from sglang.multimodal_gen.runtime.pipelines_core.lora.pipeline import LoRAPipeline


class TestPipeline(LoRAPipeline):
    def create_pipeline_stages(self, server_args):
        pass


def main() -> None:
    torch.manual_seed(7)
    device = "cuda"

    with patch(
        "sglang.multimodal_gen.runtime.layers.quantization.fp8."
        "get_tensor_model_parallel_world_size",
        return_value=1,
    ):
        base = ReplicatedLinear(
            32,
            48,
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=Fp8Config(),
        ).to(device)

    logical_weight = torch.randn(
        48, 32, device=device, dtype=torch.bfloat16
    ) / 4
    base.weight.data.copy_(logical_weight)
    base.quant_method.process_weights_after_loading(base)

    layer = BaseLayerWithLoRA(base)
    pipeline = object.__new__(TestPipeline)
    pipeline.server_args = SimpleNamespace(lora_merge_mode="auto")

    lora_a = torch.randn(8, 32, device=device, dtype=torch.bfloat16) / 8
    lora_b = torch.randn(48, 8, device=device, dtype=torch.bfloat16) / 8
    inputs = torch.randn(16, 32, device=device, dtype=torch.bfloat16)
    base_output, _ = base(inputs)
    runtime_weight_before = base.weight.detach().clone()
    layer.set_lora_weights(lora_a, lora_b, merge_weights=False)

    properties = torch.cuda.get_device_properties(0)
    print("device", torch.cuda.get_device_name(0), properties.gcnArchName)
    print(
        "logical_shape",
        tuple(logical_weight.shape),
        "runtime_shape",
        tuple(base.weight.shape),
        "scale_shape",
        tuple(base.weight_scale.shape),
        "runtime_dtype",
        base.weight.dtype,
    )

    # This directly exercises the path selected by the pre-fix auto policy.
    try:
        layer._merge_lora_into_data(base.weight.data, layer.lora_weights_list)
    except RuntimeError as error:
        print("legacy_static_merge_error", str(error).splitlines()[0])
    else:
        raise AssertionError("legacy static merge unexpectedly succeeded")
    assert torch.equal(base.weight, runtime_weight_before)

    layers = {"proj": layer}
    assert not pipeline._should_merge_lora_for_layers(
        "transformer", layers, "auto"
    )
    try:
        pipeline._should_merge_lora_for_layers("transformer", layers, "merge")
    except ValueError as error:
        print("explicit_merge_error", str(error))
    else:
        raise AssertionError("explicit merge unexpectedly admitted")
    assert torch.equal(base.weight, runtime_weight_before)

    output, _ = layer(inputs)
    reference = base_output + inputs @ lora_a.T @ lora_b.T
    difference = (output.float() - reference.float()).abs()
    print("dynamic_max_abs_vs_bf16_reference", difference.max().item())
    print("dynamic_mean_abs_vs_bf16_reference", difference.mean().item())
    assert torch.allclose(output.float(), reference.float(), atol=0.125, rtol=0.02)

    ordinary = BaseLayerWithLoRA(torch.nn.Linear(32, 48, bias=False))
    assert pipeline._should_merge_lora_for_layers(
        "ordinary", {"proj": ordinary}, "auto"
    )
    assert not pipeline._should_merge_lora_for_layers(
        "mixed", {"ordinary": ordinary, "fp8": layer}, "auto"
    )
    print("ordinary_auto_merge", True)
    print("mixed_auto_merge", False)
    print("RESULT PASS")


if __name__ == "__main__":
    main()
