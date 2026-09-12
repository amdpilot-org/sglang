"""Reproduce issue #32156's config-parsing failure without model weights.

This checks the real published checkpoint config.  It does not exercise the
NVIDIA-only NVFP4 kernel or claim a full model load on the assigned AMD GPU.
"""

import json
import sys
from pathlib import Path
from unittest import mock

import torch

from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)


def main() -> None:
    config_path = Path(sys.argv[1])
    config = json.loads(config_path.read_text())["quantization_config"]
    quant_config = CompressedTensorsConfig.from_config(config)
    layer_name = "model.language_model.layers.0.mlp.experts"
    projection_names = [
        layer_name + suffix
        for suffix in (".0.gate_proj", ".0.up_proj", ".0.down_proj")
    ]
    schemes = [
        quant_config.get_scheme_dict(torch.nn.Module(), name)
        for name in projection_names
    ]

    print(f"checkpoint_config={config_path}")
    print(f"top_level_format={quant_config.quant_format}")
    print(f"projection_matches={[scheme is not None for scheme in schemes]}")
    print(f"projection_formats={[scheme['format'] for scheme in schemes]}")
    print(
        "activation_bits="
        f"{[scheme['input_activations'].num_bits for scheme in schemes]}"
    )
    print(
        "activation_strategies="
        f"{[scheme['input_activations'].strategy for scheme in schemes]}"
    )

    selected = object()
    constructor_path = (
        "sglang.srt.layers.quantization.compressed_tensors.compressed_tensors."
        "CompressedTensorsW4A4Nvfp4MoE"
    )
    with mock.patch(constructor_path, return_value=selected) as constructor:
        result = quant_config.get_moe_scheme(torch.nn.Module(), layer_name)
    assert result is selected
    constructor.assert_called_once_with()
    print("moe_scheme=CompressedTensorsW4A4Nvfp4MoE")

    # Before PR #32736, the parser gated on top-level "mixed-precision" and
    # discarded input_activations. Replaying the reported predicate input
    # demonstrates the original exception with this checkpoint's weights.
    try:
        quant_config._is_dynamic_token_w8a8(schemes[0]["weights"], None)
    except AttributeError as exc:
        print(f"historical_failure=AttributeError: {exc}")
    else:
        raise AssertionError("historical None dereference was not reproduced")


if __name__ == "__main__":
    main()
