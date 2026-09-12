import json
from unittest import mock

import torch

from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import CompressedTensorsConfig

WEIGHTS = {"num_bits": 4, "type": "int", "symmetric": True, "strategy": "group", "group_size": 128, "dynamic": False}
TARGET = "model.layers.0.mlp.gate_proj"


def check(name, sparsity):
    config = {
        "quant_method": "compressed-tensors",
        "format": "mixed-precision",
        "config_groups": {"g": {"format": "pack-quantized", "targets": ["Linear"], "weights": WEIGHTS}},
        "sparsity_config": sparsity,
        "ignore": [],
    }
    try:
        qc = CompressedTensorsConfig.from_config(config)
        with mock.patch.object(CompressedTensorsConfig, "_check_scheme_supported", return_value=True):
            scheme = qc.get_linear_scheme(torch.nn.Linear(1, 1), TARGET)
        print(json.dumps({"case": name, "scheme": type(scheme).__name__}, sort_keys=True))
    except Exception as exc:
        print(json.dumps({"case": name, "exception": type(exc).__name__, "message": str(exc)}, sort_keys=True))


check("dense_2_4", {"format": "dense", "sparsity_structure": "2:4", "targets": ["Linear"], "ignore": []})
check("sparse_1_2", {"format": "sparse-24-bitmask", "sparsity_structure": "1:2", "targets": ["Linear"], "ignore": []})
check("target_mismatch", {"format": "sparse-24-bitmask", "sparsity_structure": "2:4", "targets": ["Conv2d"], "ignore": []})
