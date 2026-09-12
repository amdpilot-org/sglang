"""Regression coverage for Qwen3.5 split AWQ projection loading.

See https://github.com/sgl-project/sglang/issues/31720.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

from types import SimpleNamespace

import torch

from sglang.srt.layers.parameter import (
    GroupQuantScaleParameter,
    PackedvLLMParameter,
)
from sglang.srt.models.qwen3_5 import Qwen3_5GatedDeltaNet


class _MergedProjection(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.output_sizes = [16, 8, 24, 32]
        self.quant_method = SimpleNamespace(
            quant_config=SimpleNamespace(weight_block_size=(128, 128))
        )


def _recording_loader(calls):
    def loader(_param, loaded_weight, shard_id):
        calls.append((shard_id, loaded_weight.clone()))

    return loader


def test_awq_parameters_are_bound_and_split_in_checkpoint_units():
    module = _MergedProjection()
    calls = []
    original_loader = _recording_loader(calls)
    module.register_parameter(
        "qweight",
        PackedvLLMParameter(
            data=torch.empty(4, 6, dtype=torch.int32),
            input_dim=0,
            output_dim=1,
            packed_dim=1,
            packed_factor=8,
            weight_loader=original_loader,
        ),
    )
    module.register_parameter(
        "scales",
        GroupQuantScaleParameter(
            data=torch.empty(4, 48),
            input_dim=0,
            output_dim=1,
            weight_loader=original_loader,
        ),
    )

    owner = object.__new__(Qwen3_5GatedDeltaNet)
    owner._bind_packed_weight_loaders(module)

    packed = torch.arange(24, dtype=torch.int32).reshape(4, 6)
    module.qweight.weight_loader(module.qweight, packed, (0, 1, 2))
    assert [shard_id for shard_id, _ in calls] == [0, 1, 2]
    assert [chunk.shape[1] for _, chunk in calls] == [2, 1, 3]
    torch.testing.assert_close(torch.cat([chunk for _, chunk in calls], dim=1), packed)

    calls.clear()
    scales = torch.arange(192, dtype=torch.float32).reshape(4, 48)
    module.scales.weight_loader(module.scales, scales, (0, 1, 2))
    assert [chunk.shape[1] for _, chunk in calls] == [16, 8, 24]
    torch.testing.assert_close(torch.cat([chunk for _, chunk in calls], dim=1), scales)


def test_packed_split_sizes_respect_nonzero_shard_selection_and_marlin_tiles():
    module = _MergedProjection()
    loader = _recording_loader([])
    param = PackedvLLMParameter(
        data=torch.empty(1, 20, dtype=torch.int32),
        input_dim=0,
        output_dim=1,
        packed_dim=1,
        packed_factor=8,
        marlin_tile_size=2,
        weight_loader=loader,
    )

    assert Qwen3_5GatedDeltaNet._get_split_sizes_for_param(
        module, param, (1, 3)
    ) == [2, 8]


def test_packing_on_input_dimension_does_not_change_output_splits():
    module = _MergedProjection()
    loader = _recording_loader([])
    param = PackedvLLMParameter(
        data=torch.empty(1, 80, dtype=torch.int32),
        input_dim=0,
        output_dim=1,
        packed_dim=0,
        packed_factor=8,
        weight_loader=loader,
    )

    assert Qwen3_5GatedDeltaNet._get_split_sizes_for_param(
        module, param, (0, 1, 2)
    ) == [16, 8, 24]
