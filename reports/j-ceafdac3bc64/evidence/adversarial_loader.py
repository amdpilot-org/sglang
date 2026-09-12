"""Reproduce PR 1312's counterexamples using PR 1217's fixture pattern."""

from types import SimpleNamespace

import torch

from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE


for weight_width, scale_width in ((2048, 128), (1024, 64), (2048, 127)):
    loader = FusedMoE.__new__(FusedMoE)
    loader.quant_method = SimpleNamespace(load_up_proj_weight_first=True)
    loader.moe_runner_config = SimpleNamespace(is_gated=True)
    loader.moe_tp_size = 4
    loader.use_padded_loading = False
    loader.use_presharded_weights = False
    loader.use_triton_kernels = False

    weight = torch.zeros((2048, weight_width), dtype=torch.int8)
    scale = torch.zeros((2048, scale_width), dtype=torch.uint8)
    local_weight = torch.zeros((1024, weight_width), dtype=torch.int8)
    local_scale = torch.zeros((1024, scale_width), dtype=torch.uint8)
    for shard_id in ("w1", "w3"):
        loader._load_w13(local_weight, 0, shard_id, weight, 0)
        loader._load_w13(local_scale, 0, shard_id, scale, 0)
    print(f"accepted weight={tuple(weight.shape)} scale={tuple(scale.shape)}")
