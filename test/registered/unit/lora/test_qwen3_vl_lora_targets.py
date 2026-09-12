from types import SimpleNamespace

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

from sglang.srt.layers.utils import get_layer_id
from sglang.srt.lora.utils import get_hidden_dim, get_normalized_target_modules
from sglang.srt.models.qwen3_vl import Qwen3VLForConditionalGeneration


@pytest.fixture
def qwen3_vl_model():
    model = Qwen3VLForConditionalGeneration.__new__(Qwen3VLForConditionalGeneration)
    model.config = SimpleNamespace(
        vision_config=SimpleNamespace(
            hidden_size=1152,
            intermediate_size=4304,
            out_hidden_size=4096,
            spatial_merge_size=2,
        )
    )
    return model


@pytest.mark.parametrize(
    ("module_name", "expected"),
    [
        ("linear_fc1", (4608, 4608)),
        ("linear_fc2", (4608, 4096)),
    ],
)
def test_qwen3_vl_deepstack_merger_lora_dimensions(
    qwen3_vl_model, module_name, expected
):
    assert (
        get_hidden_dim(module_name, qwen3_vl_model.config, qwen3_vl_model, 0)
        == expected
    )


def test_linear_fc_targets_are_recognized():
    assert get_normalized_target_modules(["0.linear_fc1", "2.linear_fc2"]) == {
        "linear_fc1",
        "linear_fc2",
    }


@pytest.mark.parametrize("merger_index", [0, 1, 2])
def test_qwen3_vl_deepstack_adapter_weights_are_assigned_to_layers(merger_index):
    weight_name = (
        "base_model.model.model.visual.deepstack_merger_list."
        f"{merger_index}.linear_fc2.lora_A.weight"
    )
    assert get_layer_id(weight_name) == merger_index


def test_unindexed_qwen3_vl_merger_is_not_misclassified_as_a_layer():
    assert get_layer_id("model.visual.merger.linear_fc2.weight") is None
