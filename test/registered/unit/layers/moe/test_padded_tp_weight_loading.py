from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


def _loader(*, tp_size: int = 16, is_gated: bool = True) -> FusedMoE:
    loader = FusedMoE.__new__(FusedMoE)
    loader.moe_tp_size = tp_size
    loader.moe_runner_config = SimpleNamespace(is_gated=is_gated)
    loader.quant_method = SimpleNamespace(load_up_proj_weight_first=False)
    loader.quant_config = None
    loader.use_padded_loading = False
    loader.use_presharded_weights = False
    loader.use_triton_kernels = False
    return loader


@pytest.mark.parametrize("tp_rank", [0, 13, 15])
def test_load_w13_uses_unpadded_checkpoint_shard_size(tp_rank: int):
    """DeepSeek-V4-Pro has 192 source values for each padded 256 TP shard."""
    loader = _loader()
    checkpoint = torch.arange(3072)
    destination = torch.zeros(256, dtype=checkpoint.dtype)

    loader._load_w13(destination, 0, "w13", checkpoint, tp_rank)

    source_start = tp_rank * 192
    torch.testing.assert_close(
        destination[:192], checkpoint[source_start : source_start + 192]
    )
    torch.testing.assert_close(destination[192:], torch.zeros(64, dtype=torch.int64))


@pytest.mark.parametrize("tp_rank", [13, 14, 15])
def test_load_w2_scale_uses_unpadded_checkpoint_shard_size(tp_rank: int):
    """MXFP4 scales have 6 source values for each padded 8-value TP shard."""
    loader = _loader()
    checkpoint = torch.arange(96)
    destination = torch.zeros(8, dtype=checkpoint.dtype)

    loader._load_w2(destination, 0, "w2", checkpoint, tp_rank)

    source_start = tp_rank * 6
    torch.testing.assert_close(
        destination[:6], checkpoint[source_start : source_start + 6]
    )
    torch.testing.assert_close(destination[6:], torch.zeros(2, dtype=torch.int64))


def test_load_w13_unpadded_shard_fills_destination():
    loader = _loader()
    checkpoint = torch.arange(3072)
    destination = torch.full((192,), -1, dtype=checkpoint.dtype)

    loader._load_w13(destination, 0, "w13", checkpoint, tp_rank=15)

    torch.testing.assert_close(destination, checkpoint[-192:])


def test_load_split_w3_preserves_logical_half_padding():
    loader = _loader()
    checkpoint = torch.arange(3072)
    destination = torch.zeros(512, dtype=checkpoint.dtype)

    loader._load_w13(destination, 0, "w3", checkpoint, tp_rank=15)

    torch.testing.assert_close(destination[:256], torch.zeros(256, dtype=torch.int64))
    torch.testing.assert_close(destination[256:448], checkpoint[-192:])
    torch.testing.assert_close(destination[448:], torch.zeros(64, dtype=torch.int64))
