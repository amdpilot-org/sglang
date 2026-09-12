from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.utils import get_layer_id
from sglang.srt.lora.lora import LoRAAdapter
from sglang.srt.lora.lora_manager import LoRAManager
from sglang.srt.lora.utils import get_hidden_dim, get_lora_layer_id
from sglang.srt.models.moss_vl import MossVLForConditionalGeneration
from sglang.srt.models.qwen3_vl import Qwen3VLForConditionalGeneration


class Merger(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_fc1 = torch.nn.Linear(3, 5, bias=False)
        self.linear_fc2 = torch.nn.Linear(5, 2, bias=False)


class AdapterLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = {}


def make_manager(model, layer_count=6):
    manager = LoRAManager.__new__(LoRAManager)
    manager.base_model = model
    manager.base_hf_config = SimpleNamespace(num_hidden_layers=layer_count)
    manager.target_modules = {"linear_fc1", "linear_fc2"}
    wrapped = []
    manager.set_lora_module = lambda name, module: wrapped.append(name) or module
    return manager, wrapped


def test_qwen_manager_registers_each_deepstack_merger_in_its_own_slot():
    model = Qwen3VLForConditionalGeneration.__new__(Qwen3VLForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.visual = torch.nn.Module()
    model.visual.deepstack_merger_list = torch.nn.ModuleList([Merger(), Merger(), Merger()])
    manager, wrapped = make_manager(model)
    manager.init_lora_modules()
    assert len(wrapped) == 6
    for index in range(3):
        assert sorted(manager.lora_modules[index]) == [
            f"visual.deepstack_merger_list.{index}.linear_fc1",
            f"visual.deepstack_merger_list.{index}.linear_fc2",
        ]
    assert all(not manager.lora_modules[index] for index in range(3, 6))


def test_qwen_adapter_weights_follow_deepstack_indices():
    model = Qwen3VLForConditionalGeneration.__new__(Qwen3VLForConditionalGeneration)
    adapter = LoRAAdapter.__new__(LoRAAdapter)
    torch.nn.Module.__init__(adapter)
    adapter.config = SimpleNamespace(target_modules=["0.linear_fc1", "2.linear_fc2"])
    object.__setattr__(adapter, "base_model", model)
    adapter.layers = torch.nn.ModuleList([AdapterLayer() for _ in range(4)])
    for index, target in [(0, "linear_fc1"), (2, "linear_fc2")]:
        name = f"base_model.model.visual.deepstack_merger_list.{index}.{target}.lora_A.weight"
        adapter._process_weight(name, torch.tensor([index + 1.0]))
        assert name in adapter.layers[index].weights


@pytest.mark.parametrize(
    "name",
    [
        "visual.mergerish.linear_fc1",
        "visual.merger.linear_fc10",
        "notvisual.merger.linear_fc2",
        "visual.other.linear_fc2",
    ],
)
def test_moss_custom_resolver_rejects_near_miss_paths(name):
    model = MossVLForConditionalGeneration.__new__(MossVLForConditionalGeneration)
    assert get_lora_layer_id(name, model) is None


def test_generic_decoder_layer_resolution_wins_over_model_hook():
    model = MossVLForConditionalGeneration.__new__(MossVLForConditionalGeneration)
    name = "language_model.model.layers.3.mlp.down_proj"
    assert get_lora_layer_id(name, model) == 3


def test_unindexed_qwen_main_merger_remains_unrouted_by_design():
    model = Qwen3VLForConditionalGeneration.__new__(Qwen3VLForConditionalGeneration)
    assert get_layer_id("visual.merger.linear_fc1") is None
    assert get_lora_layer_id("visual.merger.linear_fc1", model) is None


def test_moss_dimensions_derive_from_deepstack_count_not_hardcoded():
    model = MossVLForConditionalGeneration.__new__(MossVLForConditionalGeneration)
    model.config = SimpleNamespace(
        vision_config=SimpleNamespace(
            hidden_size=7,
            spatial_merge_size=3,
            deepstack_visual_indexes=[2, 4],
            out_hidden_size=11,
        )
    )
    assert get_hidden_dim("linear_fc1", model.config, model, 0) == (189, 189)
    assert get_hidden_dim("linear_fc2", model.config, model, 0) == (189, 11)
