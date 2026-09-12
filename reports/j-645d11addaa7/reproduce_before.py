"""Minimal reproduction of the fusion gate at the recorded base commit."""

from sglang.srt.layers.linear import LinearBase
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

prefix = "model.layers.0.linear_attn"
projection_names = (
    "qkv_proj",
    "f_a_proj",
    "f_b_proj",
    "b_proj",
    "g_a_proj",
    "g_b_proj",
)
quant_config = Fp8Config(
    is_checkpoint_fp8_serialized=True,
    ignored_layers=[prefix],
)
probe = LinearBase(input_size=1, output_size=1, quant_config=None)
methods = {
    name: type(quant_config.get_quant_method(probe, prefix=f"{prefix}.{name}")).__name__
    for name in projection_names
}
all_source_projections_unquantized = all(
    isinstance(
        quant_config.get_quant_method(probe, prefix=f"{prefix}.{name}"),
        UnquantizedLinearMethod,
    )
    for name in projection_names
)

# Exact decision used by Glm5NextLinearAttention at the base commit.
base_gate_result = quant_config is None and 8 == 8

print("resolved_methods=", methods)
print("all_source_projections_unquantized=", all_source_projections_unquantized)
print("base_gate_result=", base_gate_result)
assert all_source_projections_unquantized
assert base_gate_result, "base gate disables fusion solely because quant_config exists"
