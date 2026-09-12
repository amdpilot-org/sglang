from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.quantization.gptq.gptq import GPTQMarlinConfig
from sglang.srt.layers.quantization.gptq.schemes.gptq_moe import (
    GPTQMarlinMoEScheme,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


def _make_scheme(desc_act: bool) -> GPTQMarlinMoEScheme:
    return GPTQMarlinMoEScheme(
        GPTQMarlinConfig(
            weight_bits=4,
            group_size=128,
            desc_act=desc_act,
            is_sym=True,
            lm_head_quantized=False,
            dynamic={},
            full_config={},
        )
    )


@pytest.mark.parametrize("params_dtype", [torch.float16, torch.bfloat16])
def test_scales_follow_params_dtype(params_dtype):
    layer = torch.nn.Module()
    layer.moe_tp_size = 1
    _make_scheme(desc_act=False).create_weights(
        layer=layer,
        num_experts=2,
        hidden_size=256,
        intermediate_size_per_partition=128,
        params_dtype=params_dtype,
    )

    assert layer.w13_scales.dtype == params_dtype
    assert layer.w2_scales.dtype == params_dtype


@pytest.mark.parametrize(
    ("desc_act", "expected_groups", "load_full"),
    [(False, 1, False), (True, 2, True)],
)
def test_w2_scale_shape_matches_tp_loading(desc_act, expected_groups, load_full):
    layer = torch.nn.Module()
    layer.moe_tp_size = 2
    _make_scheme(desc_act=desc_act).create_weights(
        layer=layer,
        num_experts=2,
        hidden_size=256,
        intermediate_size_per_partition=128,
        params_dtype=torch.bfloat16,
    )

    assert layer.w2_scales.shape == (2, expected_groups, 256)
    assert layer.w2_scales.load_full_w2 is load_full


def test_w2_loader_supports_sharded_and_full_scale_tables():
    loader = SimpleNamespace(
        quant_config=None,
        use_padded_loading=False,
        use_presharded_weights=False,
        use_triton_kernels=False,
        moe_tp_size=2,
    )
    loaded_weight = torch.arange(16, dtype=torch.float32).reshape(4, 4)

    sharded = torch.empty(2, 4)
    FusedMoE._load_w2(
        loader,
        expert_data=sharded,
        shard_dim=0,
        shard_id="w2",
        loaded_weight=loaded_weight,
        tp_rank=1,
    )
    assert torch.equal(sharded, loaded_weight[2:])

    full = torch.empty_like(loaded_weight)
    FusedMoE._load_w2(
        loader,
        expert_data=full,
        shard_dim=0,
        shard_id="w2",
        loaded_weight=loaded_weight,
        tp_rank=1,
        load_full=True,
    )
    assert torch.equal(full, loaded_weight)


@pytest.mark.parametrize("load_full", [False, True])
def test_w2_scale_table_through_weight_loader_dispatch(load_full):
    loader = FusedMoE.__new__(FusedMoE)
    torch.nn.Module.__init__(loader)
    loader.moe_tp_rank = 1
    loader.moe_tp_size = 2
    loader.quant_method = SimpleNamespace()
    loader.scheme = None
    loader.quant_config = None
    loader.use_flashinfer_trtllm_moe = False
    loader.use_triton_kernels = False
    loader.use_padded_loading = False
    loader.use_presharded_weights = False
    loader._has_fused_shared = False

    rows = 4 if load_full else 2
    param = torch.nn.Parameter(torch.full((1, rows, 4), -1.0), requires_grad=False)
    param.quant_method = "group"
    param.is_transposed = True
    param.load_full_w2 = load_full
    loaded_weight = torch.arange(16, dtype=torch.float32).reshape(4, 4)

    loader._weight_loader_impl(
        param=param,
        loaded_weight=loaded_weight,
        weight_name="experts.0.w2_scales",
        shard_id="w2",
        expert_id=0,
    )

    expected = loaded_weight if load_full else loaded_weight[2:]
    assert torch.equal(param.data[0], expected)
