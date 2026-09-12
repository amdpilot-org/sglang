"""Focused gfx950 check for compressed-tensors FP8 lm_head dispatch/scales."""

import torch

from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead


torch.manual_seed(20260912)
config = CompressedTensorsConfig.from_config(
    {
        "format": "float-quantized",
        "quant_method": "compressed-tensors",
        "ignore": [],
        "config_groups": {
            "group_0": {
                "targets": ["re:.*lm_head"],
                "weights": {
                    "num_bits": 8,
                    "type": "float",
                    "strategy": "channel",
                    "symmetric": True,
                    "dynamic": False,
                },
                "input_activations": {
                    "num_bits": 8,
                    "type": "float",
                    "strategy": "token",
                    "symmetric": True,
                    "dynamic": True,
                },
            }
        },
    }
)
head = ParallelLMHead(
    64,
    128,
    params_dtype=torch.bfloat16,
    padding_size=1,
    quant_config=config,
    prefix="lm_head",
    enable_tp=False,
)
with torch.no_grad():
    head.weight.copy_(
        (torch.randn_like(head.weight, dtype=torch.float32) * 40).to(
            torch.float8_e4m3fn
        )
    )
    head.weight_scale.copy_(torch.logspace(-4, -2, steps=64).reshape(64, 1))

head.quant_method.process_weights_after_loading(head)
head = head.cuda()
# The current ROCm fallback pads this GEMM shape to 17 tokens. Using 17 avoids
# turning this issue-specific check into a test of unrelated token padding.
hidden_states = torch.randn(17, 128, device="cuda", dtype=torch.bfloat16)
actual = head.quant_method.apply(head, hidden_states).float()
reference = hidden_states.float() @ (
    head.weight.float() * head.weight_scale.T
).float()
unscaled = hidden_states.float() @ head.weight.float()
error = (actual - reference).abs()
cosine = torch.nn.functional.cosine_similarity(
    actual.flatten(), reference.flatten(), dim=0
).item()
unscaled_cosine = torch.nn.functional.cosine_similarity(
    unscaled.flatten(), reference.flatten(), dim=0
).item()
top1_matches = (actual.argmax(-1) == reference.argmax(-1)).sum().item()

print(
    "dispatch",
    type(head.quant_method).__name__,
    "scheme",
    type(head.scheme).__name__,
    "scale_shape",
    tuple(head.weight_scale.shape),
)
print(
    "device",
    torch.cuda.get_device_name(0),
    "arch",
    torch.cuda.get_device_properties(0).gcnArchName,
)
print(
    "max_abs",
    error.max().item(),
    "rel_peak",
    error.max().item() / reference.abs().max().item(),
    "cosine",
    cosine,
)
print("top1_match", top1_matches, "/17")
print("raw_unscaled_cosine", unscaled_cosine)

assert type(head.quant_method).__name__ == "CompressedTensorsLinearMethod"
assert type(head.scheme).__name__ == "CompressedTensorsW8A8Fp8"
assert tuple(head.weight_scale.shape) == (64, 1)
assert cosine > 0.999
assert top1_matches == 17
assert unscaled_cosine < 0.8
