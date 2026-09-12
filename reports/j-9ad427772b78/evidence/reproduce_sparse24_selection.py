import json
from unittest import mock

import torch

from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)


TARGET = "model.layers.0.mlp.gate_proj"
WEIGHTS = {
    "num_bits": 4,
    "type": "int",
    "symmetric": True,
    "strategy": "group",
    "group_size": 128,
    "dynamic": False,
}
SPARSITY = {
    "format": "sparse-24-bitmask",
    "sparsity_structure": "2:4",
    "targets": ["Linear"],
    "ignore": [],
}


def run(name, config):
    quant_config = CompressedTensorsConfig.from_config(config)
    try:
        with mock.patch.object(
            CompressedTensorsConfig, "_check_scheme_supported", return_value=True
        ):
            scheme = quant_config.get_linear_scheme(torch.nn.Linear(1, 1), TARGET)
        result = {"case": name, "scheme": type(scheme).__name__}
    except Exception as exc:
        result = {"case": name, "exception": type(exc).__name__, "message": str(exc)}
    print(json.dumps(result, sort_keys=True))


base = {
    "quant_method": "compressed-tensors",
    "sparsity_config": SPARSITY,
    "ignore": [],
}

run(
    "top_level_sparse_w4a16",
    {
        **base,
        "format": "sparse-24-bitmask",
        "config_groups": {
            "group_0": {"targets": ["Linear"], "weights": WEIGHTS}
        },
    },
)
run(
    "per_group_pack_w4a16",
    {
        **base,
        "format": "mixed-precision",
        "config_groups": {
            "group_0": {
                "format": "pack-quantized",
                "targets": ["Linear"],
                "weights": WEIGHTS,
            }
        },
    },
)
run(
    "unquantized_sparse24",
    {**base, "format": "sparse-24-bitmask", "config_groups": {}},
)
