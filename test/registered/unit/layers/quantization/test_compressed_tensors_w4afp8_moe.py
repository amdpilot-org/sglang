# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
from compressed_tensors import CompressionFormat

from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a8_fp8_moe import (
    CompressedTensorsW4AFP8MoE,
)


def make_quant_config(group_size: int):
    weight_config = SimpleNamespace(
        num_bits=4,
        group_size=group_size,
        symmetric=True,
    )
    return SimpleNamespace(
        target_scheme_map={"Linear": {"weights": weight_config}},
        quant_format=CompressionFormat.pack_quantized.value,
    )


def test_group_size_128_is_supported():
    scheme = CompressedTensorsW4AFP8MoE(make_quant_config(128), None, None)

    assert scheme.group_size == 128


@pytest.mark.parametrize("group_size", [64, 127, 256])
def test_unsupported_group_size_is_rejected(group_size):
    with pytest.raises(ValueError, match=r"only supports group_size=128"):
        CompressedTensorsW4AFP8MoE(make_quant_config(group_size), None, None)
