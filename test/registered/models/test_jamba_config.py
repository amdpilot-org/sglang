from sglang.srt.configs.jamba import JambaConfig
from sglang.srt.models.registry import ModelRegistry
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=5, suite="stage-b-test-1-gpu-small-amd")


def test_jamba_15_layer_pattern_and_cache_shape():
    config = JambaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=8,
        num_attention_heads=8,
        num_key_value_heads=2,
        attn_layer_period=4,
        attn_layer_offset=2,
        expert_layer_period=2,
        expert_layer_offset=1,
        num_experts=16,
        mamba_expand=2,
        mamba_d_state=16,
        mamba_d_conv=4,
    )
    assert config.full_attention_layer_ids == [2, 6]
    assert config.mamba_layer_ids == [0, 1, 3, 4, 5, 7]
    with get_parallel().override(attn_tp_size=2):
        params = config.mamba2_cache_params
    assert params.layers == config.mamba_layer_ids
    assert params.shape.conv == [(64, 3)]
    assert params.shape.temporal == (64, 16)


def test_jamba_architecture_is_registered_natively():
    model_cls, architecture = ModelRegistry.resolve_model_cls(["JambaForCausalLM"])
    assert architecture == "JambaForCausalLM"
    assert model_cls.__module__ == "sglang.srt.models.jamba"
