"""DeepSeek-V4 packed-MXFP4 routed-expert loader regression tests."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.quantization.fp8 import Fp8MoEMethod
from sglang.srt.layers.quantization.mxfp4_flashinfer_cutlass_moe import (
    Mxfp4FlashinferCutlassMoEMethod,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.moe_runner_config = SimpleNamespace(is_gated=True)


def _make_loader(hidden_size=4096, intermediate_size=1024):
    """Create the buffers through the production DSv4 SM90 quant contract."""
    quant_config = SimpleNamespace(
        activation_scheme="dynamic",
        is_checkpoint_fp8_serialized=True,
        is_fp4_experts=True,
        weight_block_size=(128, 128),
    )
    fp8_method = Fp8MoEMethod.__new__(Fp8MoEMethod)
    fp8_method.block_quant = True
    fp8_method.quant_config = quant_config
    fp8_method.use_mxfp8 = False
    fp8_method.is_fp4_expert = True

    method = Mxfp4FlashinferCutlassMoEMethod.__new__(
        Mxfp4FlashinferCutlassMoEMethod
    )
    method._fp8 = fp8_method
    layer = _Layer()

    # Make this host-independent: the contract under test is the CUDA SM90
    # method, while the assigned CI host may have ROCm/AITER enabled.
    with (
        patch("sglang.srt.layers.quantization.fp8._use_aiter", False),
        patch(
            "sglang.srt.layers.quantization.fp8.get_parallel",
            return_value=SimpleNamespace(tp_size=4),
        ),
    ):
        method.create_weights(
            layer,
            num_experts=1,
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size,
            params_dtype=torch.bfloat16,
        )

    loader = FusedMoE.__new__(FusedMoE)
    loader.quant_method = method
    loader.quant_config = quant_config
    loader.moe_runner_config = layer.moe_runner_config
    loader.moe_tp_size = 4
    loader.use_padded_loading = False
    loader.use_presharded_weights = False
    loader.use_triton_kernels = False
    return loader, layer


@pytest.mark.parametrize("tp_rank", range(4))
def test_reported_mxfp4_weight_and_scale_shapes_shard_at_tp4(tp_rank):
    """Load the issue's tensors into buffers derived from hidden_size=4096."""
    loader, layer = _make_loader()
    checkpoint_weight = torch.arange(2048, dtype=torch.int16)[:, None].expand(
        2048, 2048
    ).to(torch.int8)
    checkpoint_scale = (
        (torch.arange(2048, dtype=torch.int16) % 4 + 125)
        .to(torch.uint8)[:, None]
        .expand(2048, 128)
        .view(torch.float8_e8m0fnu)
    )
    shard = slice(tp_rank * 512, (tp_rank + 1) * 512)

    for shard_id, destination in (("w1", 1024), ("w3", 0)):
        loader._load_w13(
            layer.w13_weight.data[0], 0, shard_id, checkpoint_weight, tp_rank
        )
        loader._load_w13(
            layer.w13_weight_scale_inv.data[0],
            0,
            shard_id,
            checkpoint_scale,
            tp_rank,
        )
        assert torch.equal(
            layer.w13_weight.data[0, destination : destination + 512],
            checkpoint_weight[shard],
        )
        assert torch.equal(
            layer.w13_weight_scale_inv.data[
                0, destination : destination + 512
            ].view(torch.uint8),
            checkpoint_scale[shard].view(torch.uint8),
        )


@pytest.mark.parametrize(
    "weight_width,scale_width",
    [(1024, 64), (2048, 127)],
)
def test_counterexample_widths_do_not_match_reported_model_contract(
    weight_width, scale_width
):
    _, layer = _make_loader(hidden_size=4096)
    checkpoint_weight = torch.zeros((2048, weight_width), dtype=torch.int8)
    checkpoint_scale = torch.zeros(
        (2048, scale_width), dtype=torch.uint8
    ).view(torch.float8_e8m0fnu)

    expected_weight = layer.w13_weight.shape[1:]
    expected_scale = layer.w13_weight_scale_inv.shape[1:]
    assert (checkpoint_weight.shape != expected_weight) or (
        checkpoint_scale.shape != expected_scale
    )


def test_other_aligned_hidden_size_has_its_own_valid_scale_width():
    """Do not mistake a different valid model shape for malformed MXFP4."""
    _, layer = _make_loader(hidden_size=2048)
    assert layer.w13_weight.shape == (1, 2048, 1024)
    assert layer.w13_weight_scale_inv.shape == (1, 2048, 64)
