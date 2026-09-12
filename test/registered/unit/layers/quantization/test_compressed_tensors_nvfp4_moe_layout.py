from unittest.mock import Mock

import pytest
import torch

from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsFusedMoEMethod,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_scheme import (
    CompressedTensorsMoEScheme,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a4_nvfp4_moe import (
    CompressedTensorsW4A4Nvfp4MoE,
)


@pytest.mark.parametrize(
    ("use_flashinfer_trtllm", "expected"),
    [(False, True), (True, False)],
)
def test_nvfp4_moe_w13_layout_matches_backend(use_flashinfer_trtllm, expected):
    scheme = object.__new__(CompressedTensorsW4A4Nvfp4MoE)
    scheme.use_flashinfer_trtllm = use_flashinfer_trtllm

    assert scheme.load_up_proj_weight_first is expected


def test_compressed_tensors_moe_default_keeps_gate_up_layout():
    assert CompressedTensorsMoEScheme.load_up_proj_weight_first is False


def test_compressed_tensors_moe_forwards_w13_layout_to_loader():
    method = object.__new__(CompressedTensorsFusedMoEMethod)
    scheme = Mock(load_up_proj_weight_first=True)
    layer = Mock(scheme=scheme)

    method.create_weights(
        layer=layer,
        num_experts=2,
        hidden_size=32,
        intermediate_size_per_partition=16,
        params_dtype=torch.bfloat16,
    )

    assert method.load_up_proj_weight_first is True
    scheme.create_weights.assert_called_once()
