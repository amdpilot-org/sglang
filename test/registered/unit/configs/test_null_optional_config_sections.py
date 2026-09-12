import pytest

from sglang.srt.configs.muse_glimmer import muse_glimmer_config_kwargs_from_hf
from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp8Config
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


@pytest.mark.parametrize(
    "config",
    [
        {"model_type": "muse_glimmer"},
        {"model_type": "muse_glimmer", "vision_config": {"sentinel": True}},
    ],
)
def test_muse_glimmer_null_text_config_matches_missing_section(config):

    assert muse_glimmer_config_kwargs_from_hf(
        {**config, "text_config": None}
    ) == muse_glimmer_config_kwargs_from_hf(config)


def test_modelopt_fp8_null_quantization_matches_missing_section():
    expected_message = "Cannot find 'quant_algo'"

    with pytest.raises(ValueError, match=expected_message):
        ModelOptFp8Config.from_config({})
    with pytest.raises(ValueError, match=expected_message):
        ModelOptFp8Config.from_config({"quantization": None})


def test_modelopt_fp8_valid_nested_quantization_is_preserved():
    config = ModelOptFp8Config.from_config(
        {
            "quantization": {
                "quant_algo": "FP8",
                "kv_cache_quant_algo": "FP8",
                "exclude_modules": ["lm_head"],
            }
        }
    )

    assert config.kv_cache_quant_algo == "FP8"
    assert config.exclude_modules == ["lm_head"]
