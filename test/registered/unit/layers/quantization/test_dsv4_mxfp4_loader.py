"""DeepSeek-V4 packed-MXFP4 routed-expert loader regression tests."""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


@pytest.mark.parametrize("tp_rank", range(4))
def test_reported_mxfp4_weight_and_scale_shapes_shard_at_tp4(tp_rank):
    """Load the issue's [2048, 2048] / [2048, 128] expert tensors.

    MXFP4 packs two values per int8 byte, so the checkpoint's second weight
    dimension represents hidden_size=4096.  TP shards the intermediate/output
    dimension, while the packed hidden dimension and its 128 E8M0 groups stay
    intact.  FlashInfer consumes W13 in [up; gate] order.
    """
    checkpoint_weight = torch.arange(2048, dtype=torch.int16)[:, None].expand(
        2048, 2048
    )
    checkpoint_weight = checkpoint_weight.to(torch.int8)
    checkpoint_scale = torch.arange(2048, dtype=torch.int16)[:, None].expand(
        2048, 128
    )
    checkpoint_scale = checkpoint_scale.to(torch.uint8)

    loader = FusedMoE.__new__(FusedMoE)
    loader.quant_method = SimpleNamespace(load_up_proj_weight_first=True)
    loader.moe_runner_config = SimpleNamespace(is_gated=True)
    loader.moe_tp_size = 4
    loader.use_padded_loading = False
    loader.use_presharded_weights = False
    loader.use_triton_kernels = False

    local_weight = torch.zeros((1024, 2048), dtype=torch.int8)
    local_scale = torch.zeros((1024, 128), dtype=torch.uint8)
    shard = slice(tp_rank * 512, (tp_rank + 1) * 512)

    for shard_id, destination in (("w1", 512), ("w3", 0)):
        loader._load_w13(
            expert_data=local_weight,
            shard_dim=0,
            shard_id=shard_id,
            loaded_weight=checkpoint_weight,
            tp_rank=tp_rank,
        )
        loader._load_w13(
            expert_data=local_scale,
            shard_dim=0,
            shard_id=shard_id,
            loaded_weight=checkpoint_scale,
            tp_rank=tp_rank,
        )

        assert torch.equal(
            local_weight[destination : destination + 512],
            checkpoint_weight[shard],
        )
        assert torch.equal(
            local_scale[destination : destination + 512],
            checkpoint_scale[shard],
        )
