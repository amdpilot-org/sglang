# SPDX-License-Identifier: Apache-2.0
"""Regression tests for diffusion disk/server checksum comparison."""

from pathlib import Path

import torch
from safetensors.torch import save_file

from sglang.multimodal_gen.runtime.models.parameter import BasevLLMParameter
from sglang.multimodal_gen.runtime.post_training.gpu_worker_post_training_mixin import (
    GPUWorkerPostTrainingMixin,
)
from sglang.multimodal_gen.runtime.post_training.weights_updater import (
    _get_weights_iter,
    _load_weights_into_module,
    compare_module_weights_with_disk,
)


class _MappedTextEncoder(torch.nn.Module):
    param_names_mapping = {
        r"^encoder\.renamed\.": r"renamed.",
        r"^encoder\.q_proj\.": (r"qkv.", 0, 3),
        r"^encoder\.k_proj\.": (r"qkv.", 1, 3),
        r"^encoder\.v_proj\.": (r"qkv.", 2, 3),
    }

    def __init__(self):
        super().__init__()
        self.renamed = torch.nn.Linear(2, 2, bias=False)
        self.qkv = torch.nn.Linear(2, 6, bias=False)

        def load_qkv(param, value, shard_id):
            width = param.shape[0] // 3
            param.data.narrow(0, shard_id * width, width).copy_(value)

        self.qkv.weight.weight_loader = load_qkv


def _write_checkpoint(path: Path) -> None:
    path.mkdir()
    save_file(
        {
            "encoder.renamed.weight": torch.arange(4, dtype=torch.float32).view(2, 2),
            "encoder.q_proj.weight": torch.full((2, 2), 1.0),
            "encoder.k_proj.weight": torch.full((2, 2), 2.0),
            "encoder.v_proj.weight": torch.full((2, 2), 3.0),
            # Real checkpoints may contain values intentionally ignored by a loader.
            "encoder.runtime_only.weight": torch.ones(1),
        },
        path / "model.safetensors",
    )


def test_comparison_replays_name_mapping_and_fused_loader(tmp_path):
    checkpoint = tmp_path / "text_encoder"
    _write_checkpoint(checkpoint)
    module = _MappedTextEncoder()
    _load_weights_into_module(module, _get_weights_iter(str(checkpoint)))

    result = compare_module_weights_with_disk(module, str(checkpoint))

    assert result["match"] is True
    assert result["server_checksum"] == result["disk_checksum"]
    assert result["parameter_count"] == 2


def test_comparison_detects_single_value_corruption(tmp_path):
    checkpoint = tmp_path / "vae"
    _write_checkpoint(checkpoint)
    module = _MappedTextEncoder()
    _load_weights_into_module(module, _get_weights_iter(str(checkpoint)))
    module.qkv.weight.data[5, 1].nextafter_(torch.tensor(float("inf")))

    result = compare_module_weights_with_disk(module, str(checkpoint))

    assert result["match"] is False
    assert result["server_checksum"] != result["disk_checksum"]


class _ColumnParallelTextEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

        def load_column_parallel(param, loaded_weight):
            param.load_column_parallel_weight(loaded_weight)

        self.weight = BasevLLMParameter(
            torch.zeros(2), weight_loader=load_column_parallel
        )


def test_comparison_preserves_parameter_subclass_loader_api(tmp_path):
    checkpoint = tmp_path / "text_encoder"
    checkpoint.mkdir()
    save_file(
        {"weight": torch.tensor([5.0, 6.0])}, checkpoint / "model.safetensors"
    )
    module = _ColumnParallelTextEncoder()

    _load_weights_into_module(module, _get_weights_iter(str(checkpoint)))
    result = compare_module_weights_with_disk(module, str(checkpoint))

    assert module.weight.tolist() == [5.0, 6.0]
    assert result["match"] is True
    module.weight.data[1] = 7.0
    assert compare_module_weights_with_disk(module, str(checkpoint))["match"] is False


def test_comparison_rejects_checkpoint_with_no_loadable_parameters(tmp_path):
    checkpoint = tmp_path / "text_encoder"
    checkpoint.mkdir()
    save_file({"unknown.weight": torch.ones(2)}, checkpoint / "model.safetensors")

    module = _MappedTextEncoder()
    try:
        compare_module_weights_with_disk(module, str(checkpoint))
    except ValueError as exc:
        assert "No checkpoint state matched" in str(exc)
    else:
        raise AssertionError("empty comparisons must not be reported as matching")


class _VaeWithRunningState(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0]))
        self.register_buffer("running_mean", torch.tensor([9.0]))


def test_comparison_detects_checkpoint_buffer_corruption(tmp_path):
    checkpoint = tmp_path / "vae"
    checkpoint.mkdir()
    save_file(
        {"weight": torch.tensor([1.0]), "running_mean": torch.tensor([2.0])},
        checkpoint / "model.safetensors",
    )

    result = compare_module_weights_with_disk(_VaeWithRunningState(), str(checkpoint))

    assert result["match"] is False
    assert result["server_checksum"] != result["disk_checksum"]
    assert result["parameter_count"] == 1
    assert result["buffer_count"] == 1
    assert result["tensor_count"] == 2


def test_explicit_empty_module_selection_is_rejected(tmp_path):
    class _Pipeline:
        modules = {"vae": _VaeWithRunningState()}

    class _Worker(GPUWorkerPostTrainingMixin):
        pipeline = _Pipeline()

    result = _Worker().compare_weights_with_disk(str(tmp_path), module_names=[])

    assert result == {
        "success": False,
        "message": "At least one module must be selected",
        "modules": {},
    }
