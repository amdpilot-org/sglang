from types import SimpleNamespace

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

from sglang.srt.lora.utils import get_hidden_dim
from sglang.srt.models.moss_vl import MossVLForConditionalGeneration


@pytest.fixture
def moss_vl_model():
    text_config = SimpleNamespace(
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=8,
        intermediate_size=11008,
    )
    model = MossVLForConditionalGeneration.__new__(MossVLForConditionalGeneration)
    model.config = SimpleNamespace(
        vision_config=SimpleNamespace(
            hidden_size=1024,
            out_hidden_size=4096,
            spatial_merge_size=2,
            deepstack_visual_indexes=[7, 15, 23],
        ),
        get_text_config=lambda: text_config,
    )
    return model


@pytest.mark.parametrize(
    ("module_name", "expected"),
    [
        ("linear_fc1", (16384, 16384)),
        ("linear_fc2", (16384, 4096)),
    ],
)
def test_moss_vl_concatenating_merger_lora_dimensions(
    moss_vl_model, module_name, expected
):
    assert (
        get_hidden_dim(module_name, moss_vl_model.config, moss_vl_model, 0) == expected
    )


def test_moss_vl_text_target_delegates_to_default_dimensions(moss_vl_model):
    assert get_hidden_dim("down_proj", moss_vl_model.config, moss_vl_model, 0) == (
        11008,
        4096,
    )
