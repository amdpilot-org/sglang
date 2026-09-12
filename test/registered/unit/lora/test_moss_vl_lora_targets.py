from types import SimpleNamespace

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

from sglang.srt.lora.lora import LoRAAdapter
from sglang.srt.lora.lora_manager import LoRAManager
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


class _Merger(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_fc1 = torch.nn.Linear(4, 4, bias=False)
        self.linear_fc2 = torch.nn.Linear(4, 2, bias=False)


class _AdapterLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = {}


def test_moss_vl_unindexed_merger_modules_are_registered(moss_vl_model):
    torch.nn.Module.__init__(moss_vl_model)
    moss_vl_model.visual = torch.nn.Module()
    moss_vl_model.visual.merger = _Merger()

    manager = LoRAManager.__new__(LoRAManager)
    manager.base_model = moss_vl_model
    manager.base_hf_config = SimpleNamespace(num_hidden_layers=4)
    manager.target_modules = {"linear_fc1", "linear_fc2"}
    wrapped = []
    manager.set_lora_module = lambda name, module: wrapped.append(name) or module

    manager.init_lora_modules()

    expected = ["visual.merger.linear_fc1", "visual.merger.linear_fc2"]
    assert wrapped == expected
    assert sorted(manager.lora_modules[0]) == expected
    assert all(not modules for modules in manager.lora_modules[1:])


@pytest.mark.parametrize("module_name", ["linear_fc1", "linear_fc2"])
def test_moss_vl_unindexed_merger_adapter_weights_use_layer_zero(
    moss_vl_model, module_name
):
    adapter = LoRAAdapter.__new__(LoRAAdapter)
    torch.nn.Module.__init__(adapter)
    adapter.config = SimpleNamespace(target_modules=[module_name])
    object.__setattr__(adapter, "base_model", moss_vl_model)
    adapter.layers = torch.nn.ModuleList([_AdapterLayer() for _ in range(4)])
    weight_name = f"base_model.model.visual.merger.{module_name}.lora_A.weight"
    weight = torch.ones(1, 1)

    adapter._process_weight(weight_name, weight)

    assert adapter.layers[0].weights[weight_name].equal(weight)
    assert all(not layer.weights for layer in adapter.layers[1:])
