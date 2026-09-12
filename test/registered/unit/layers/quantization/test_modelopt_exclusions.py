from unittest.mock import patch

from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp4Config
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod


GLM_PACKED_MODULES = {
    "fused_qkvbfg_a_proj": [
        "q_proj",
        "k_proj",
        "v_proj",
        "b_proj",
        "f_a_proj",
        "g_a_proj",
    ]
}


def make_config(exclude_modules, packed_modules_mapping=GLM_PACKED_MODULES):
    return ModelOptFp4Config(
        is_checkpoint_nvfp4_serialized=True,
        group_size=16,
        exclude_modules=exclude_modules,
        packed_modules_mapping=packed_modules_mapping,
    )


def test_glm_fused_module_honors_checkpoint_shard_exclusion():
    config = make_config(["model.layers.0.self_attn.q_proj"])
    prefix = "model.layers.0.self_attn.fused_qkvbfg_a_proj"

    assert config.is_layer_excluded(prefix)

    linear = ReplicatedLinear.__new__(ReplicatedLinear)
    with patch(
        "sglang.srt.layers.quantization.modelopt_quant.is_layer_skipped",
        side_effect=AssertionError("generic mixed-shard validation must be bypassed"),
    ):
        assert isinstance(config.get_quant_method(linear, prefix), UnquantizedLinearMethod)


def test_glm_renamed_visual_prefix_honors_model_prefixed_exclusion():
    config = make_config(
        ["model.visual.blocks.*.attn.qkv"], packed_modules_mapping={}
    )

    assert config.is_layer_excluded("visual.blocks.0.attn.qkv")


def test_unrelated_fused_shard_and_similar_prefix_remain_quantized():
    config = make_config(["model.layers.0.self_attn.o_proj"])

    assert not config.is_layer_excluded(
        "model.layers.0.self_attn.fused_qkvbfg_a_proj"
    )
    assert not config.is_layer_excluded("visualizer.blocks.0.attn.qkv")


def test_existing_language_model_prefix_normalization_is_preserved():
    config = make_config(["model.layers.0.mlp.down_proj"])

    assert config.is_layer_excluded("language_model.model.layers.0.mlp.down_proj")
