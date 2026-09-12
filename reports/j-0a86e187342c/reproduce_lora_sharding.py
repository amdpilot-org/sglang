"""Deterministic evidence for issue 35415's fused LoRA-B TP slicing."""

from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.multimodal_gen.runtime.layers.lora.linear import (
    MergedColumnParallelLinearWithLoRA,
)


TP_RANK = "sglang.multimodal_gen.runtime.layers.lora.linear.get_tp_rank"


def legacy_slice(layer, tensor, rank):
    """Implementation from before upstream PR 33875."""
    shard_size = layer.base_layer.output_partition_sizes[0]
    start = rank * shard_size
    return tensor[:, start : start + shard_size, :]


class FakeMergedLinear(torch.nn.Module):
    def __init__(self, output_sizes, partition_sizes):
        super().__init__()
        self.output_sizes = output_sizes
        self.output_partition_sizes = partition_sizes
        self.weight = torch.nn.Parameter(torch.empty(sum(partition_sizes), 1))
        self.bias = None
        self.skip_bias_add = False
        self.gather_output = False
        self.quant_method = SimpleNamespace()


def main():
    device = torch.device("cuda")
    output_sizes = [8, 2, 2]
    partition_sizes = [4, 1, 1]
    layer = MergedColumnParallelLinearWithLoRA(
        FakeMergedLinear(output_sizes, partition_sizes)
    ).to(device)
    fused_2d = torch.arange(24, device=device).reshape(12, 2)

    try:
        legacy_slice(layer, fused_2d, 0)
    except IndexError as error:
        print(f"legacy_2d_failure={type(error).__name__}: {error}")
    else:
        raise AssertionError("legacy 3-D indexing unexpectedly accepted a 2-D tensor")

    expected_rows = {
        0: torch.tensor([0, 1, 2, 3, 8, 10], device=device),
        1: torch.tensor([4, 5, 6, 7, 9, 11], device=device),
    }
    for rank in (0, 1):
        with patch(TP_RANK, return_value=rank):
            actual = layer.slice_lora_b_weights(fused_2d)
        expected = fused_2d.index_select(0, expected_rows[rank])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        print(f"current_2d_rank_{rank}_rows={expected_rows[rank].cpu().tolist()}")

    stacked_3d = torch.arange(3 * 8 * 2, device=device).reshape(3, 8, 2)
    for rank in (0, 1):
        with patch(TP_RANK, return_value=rank):
            actual = layer.slice_lora_b_weights(stacked_3d)
        expected = stacked_3d[:, rank * 4 : (rank + 1) * 4, :]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        print(f"current_3d_rank_{rank}_shape={list(actual.shape)}")

    torch.cuda.synchronize()
    print(f"device={torch.cuda.get_device_name(0)}")
    print("status=pass")


if __name__ == "__main__":
    main()
