from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.lora.backend.torch_backend import TorchNativeLoRABackend
from sglang.srt.lora.layers import (
    ColumnParallelLinearWithLoRA,
    RowParallelLinearWithLoRA,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _DecodeMode:
    def is_decode(self):
        return True

    def is_target_verify(self):
        return False

    def is_extend(self):
        return False


class _Linear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(3, 4))
        self.output_partition_sizes = [3]
        self.output_size = 3


class TestTorchNativeLoRACPU(CustomTestCase):
    def setUp(self):
        self.backend = TorchNativeLoRABackend(
            max_loras_per_batch=2, device=torch.device("cpu")
        )
        self.batch = SimpleNamespace(forward_mode=_DecodeMode(), batch_size=2)

    def test_cpu_layer_offsets_do_not_require_pinned_allocator(self):
        base_layer = _Linear()
        with patch.object(
            torch.Tensor,
            "pin_memory",
            side_effect=RuntimeError("no pinned memory allocator is available"),
        ):
            column = ColumnParallelLinearWithLoRA(base_layer, self.backend)
            row = RowParallelLinearWithLoRA(base_layer, self.backend)
            row.set_lora_info(torch.empty(0), torch.empty(0))

        self.assertEqual(column.output_offset_cpu.device.type, "cpu")
        self.assertEqual(column.output_offset_cpu.tolist(), [0, 3])
        self.assertEqual(row.output_offset_cpu.tolist(), [0, 3])
        self.assertFalse(column.output_offset_cpu.is_pinned())
        self.assertFalse(row.output_offset_cpu.is_pinned())

    def test_cpu_batch_metadata_does_not_require_pinned_allocator(self):
        with patch.object(
            torch.Tensor,
            "pin_memory",
            side_effect=RuntimeError("no pinned memory allocator is available"),
        ):
            self.backend.prepare_lora_batch(
                self.batch,
                weight_indices=[0, 1],
                lora_ranks=[2, 2],
                scalings=[0.5, 2.0],
                use_cuda_graph=False,
            )

        info = self.backend.batch_info
        self.assertEqual(info.seg_lens_cpu.tolist(), [1, 1])
        self.assertEqual(info.seg_indptr_cpu.tolist(), [0, 1, 2])
        self.assertEqual(info.weight_indices_cpu.tolist(), [0, 1])
        self.assertFalse(info.seg_lens_cpu.is_pinned())
        self.assertFalse(info.weight_indices_cpu.is_pinned())

    def test_cpu_output_matches_independent_lora_reference(self):
        self.backend.prepare_lora_batch(
            self.batch,
            weight_indices=[0, 1],
            lora_ranks=[2, 2],
            scalings=[0.5, 2.0],
            use_cuda_graph=False,
        )
        inputs = torch.tensor([[1.0, 2.0, -1.0], [-2.0, 0.5, 3.0]])
        lora_a = torch.tensor(
            [
                [[1.0, 0.0, 2.0], [0.0, -1.0, 1.0]],
                [[2.0, 1.0, 0.0], [-1.0, 0.0, 0.5]],
            ]
        )
        lora_b = torch.tensor(
            [
                [[1.0, 2.0], [0.0, -1.0], [3.0, 0.5]],
                [[-1.0, 0.0], [2.0, 1.0], [0.5, -2.0]],
            ]
        )

        hidden = self.backend.run_lora_a_sgemm(inputs, lora_a)
        actual = self.backend.run_lora_b_sgemm(
            hidden, lora_b, torch.tensor([0, 3], dtype=torch.int32)
        )
        expected = torch.stack(
            [
                0.5 * (lora_b[0] @ (lora_a[0] @ inputs[0])),
                2.0 * (lora_b[1] @ (lora_a[1] @ inputs[1])),
            ]
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    import unittest

    unittest.main(verbosity=2)
