# Adapted from https://github.com/thinking-machines-lab/batch_invariant_ops/blob/main/test_batch_invariance.py
import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.batch_invariant_ops import batch_invariant_ops
from sglang.srt.batch_invariant_ops.batch_invariant_ops import set_batch_invariant_mode
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

# Note: MI300 (gfx942) has 64KB shared memory limit but kernel needs 66KB
# MI35x (gfx950/CDNA4) may have different limits - testing on MI35x only
register_cuda_ci(est_time=20, stage="nightly", runner_config="1-gpu-large")
register_amd_ci(est_time=10, suite="nightly-amd-1-gpu-mi35x", nightly=True)

device_type = getattr(torch.accelerator.current_accelerator(), "type", "cpu")
torch.set_default_device(device_type)

# Just to get the logging out of the way
with set_batch_invariant_mode(True):
    pass


class TestBatchInvariantOps(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        batch_invariant_ops._ENABLE_MM_COMPARISON_TEST = True

    @classmethod
    def tearDownClass(cls):
        batch_invariant_ops._ENABLE_MM_COMPARISON_TEST = False

    def _test_batch_invariance(self, M, K, N, dtype):
        """
        Test that matrix operations produce identical results for:
        - Method 1: Matrix-vector multiplication (batch size 1)
        - Method 2: Matrix-matrix multiplication, then slice (full batch)
        """
        a = torch.linspace(-100, 100, M * K, dtype=dtype).reshape(M, K)

        # Create non-contiguous tensor
        b = torch.linspace(-100, 100, K * N, dtype=dtype).reshape(N, K)
        b = b.transpose(0, 1)

        # Method 1: Matrix-vector multiplication (batch size 1)
        out1 = torch.mm(a[:1], b)

        # Method 2: Matrix-matrix multiplication, then slice (full batch)
        out2_pre = torch.mm(a, b)
        out2 = out2_pre[:1]

        # Check if results are identical
        diff = (out1 - out2).abs().max()
        return diff.item()

    def _run_multiple_iterations(self, iters, M, K, N, dtype):
        """Run multiple iterations and collect diff statistics"""
        difflist = []
        for _ in range(iters):
            diff = self._test_batch_invariance(M, K, N, dtype)
            difflist.append(diff)
        return difflist

    def _assert_batch_invariant_results(self, difflist, dtype, test_name):
        """
        Assert that in batch-invariant mode:
        1. All diffs must not be NaN
        2. All diffs must be exactly 0
        3. Max, min, and diff of diffs must all be 0
        """
        max_diff = max(difflist)
        min_diff = min(difflist)
        diff_range = max_diff - min_diff

        # Check for NaN values
        self.assertFalse(
            math.isnan(max_diff), f"{test_name}: max_diff is NaN for {dtype}"
        )
        self.assertFalse(
            math.isnan(min_diff), f"{test_name}: min_diff is NaN for {dtype}"
        )
        self.assertFalse(
            math.isnan(diff_range), f"{test_name}: diff_range is NaN for {dtype}"
        )

        # Check that all diffs are exactly 0
        self.assertEqual(
            max_diff,
            0.0,
            f"{test_name}: max_diff must be 0 in batch-invariant mode, got {max_diff} for {dtype}",
        )
        self.assertEqual(
            min_diff,
            0.0,
            f"{test_name}: min_diff must be 0 in batch-invariant mode, got {min_diff} for {dtype}",
        )
        self.assertEqual(
            diff_range,
            0.0,
            f"{test_name}: diff_range must be 0 in batch-invariant mode, got {diff_range} for {dtype}",
        )

    def test_small_matrices(self):
        """Test batch invariance with small matrix sizes"""
        test_cases = [
            ("Small-1", 8, 64, 128),
            ("Small-2", 16, 128, 256),
            ("Small-3", 4, 32, 64),
        ]

        for name, M, K, N in test_cases:
            with self.subTest(name=name, M=M, K=K, N=N):
                for dtype in [torch.float32, torch.bfloat16]:
                    with self.subTest(dtype=dtype):
                        # Run with batch-invariant mode
                        with set_batch_invariant_mode(True):
                            difflist = self._run_multiple_iterations(
                                iters=5, M=M, K=K, N=N, dtype=dtype
                            )
                            self._assert_batch_invariant_results(difflist, dtype, name)

    def test_mm_dtype_fp32_preserves_accumulator_precision(self):
        torch.manual_seed(34758)
        a = torch.randn((5, 257), dtype=torch.bfloat16)
        b = torch.randn((257, 64), dtype=torch.bfloat16)

        with set_batch_invariant_mode(True):
            actual = torch.mm(a, b, out_dtype=torch.float32)

        reference = (a.cpu().double() @ b.cpu().double()).float().to(actual.device)
        widened_bf16 = actual.bfloat16().float()
        self.assertEqual(actual.dtype, torch.float32)
        self.assertGreater(torch.count_nonzero(actual != widened_bf16).item(), 0)
        self.assertLess(
            torch.max(torch.abs(actual - reference)).item(),
            torch.max(torch.abs(widened_bf16 - reference)).item(),
        )

    def test_mm_dtype_fp32_is_repeatable_and_sub_batch_invariant(self):
        torch.manual_seed(34758)
        a = torch.randn((17, 257), dtype=torch.bfloat16)
        b = torch.randn((257, 64), dtype=torch.bfloat16)

        with set_batch_invariant_mode(True):
            full = torch.mm(a, b, out_dtype=torch.float32)
            repeated = torch.mm(a, b, out_dtype=torch.float32)
            sub_batch = torch.mm(a[:5], b, out_dtype=torch.float32)

        self.assertTrue(torch.equal(full, repeated))
        self.assertTrue(torch.equal(full[:5], sub_batch))

    def test_mm_dtype_fp32_preserves_near_tied_expert_order(self):
        hidden_states = torch.zeros((1, 257), dtype=torch.bfloat16)
        weights = torch.zeros((257, 64), dtype=torch.bfloat16)
        hidden_states[0, :2] = 1
        weights[0, :2] = 100
        weights[1, 1] = 0.0625

        with set_batch_invariant_mode(True):
            scores = torch.mm(hidden_states, weights, out_dtype=torch.float32)

        self.assertEqual(scores[0, 0].item(), 100.0)
        self.assertEqual(scores[0, 1].item(), 100.0625)
        self.assertEqual(scores.bfloat16()[0, 0].item(), 100.0)
        self.assertEqual(scores.bfloat16()[0, 1].item(), 100.0)
        self.assertEqual(torch.argmax(scores).item(), 1)
        self.assertEqual(torch.argmax(scores.bfloat16()).item(), 0)

    def test_fp32_output_bypasses_bf16_only_deepgemm(self):
        a = torch.empty((1, 16), dtype=torch.bfloat16)
        b = torch.empty((16, 16), dtype=torch.bfloat16)
        expected = torch.empty((1, 16), dtype=torch.float32)

        with (
            patch.object(batch_invariant_ops, "_ENABLE_MM_DEEPGEMM", True),
            patch.object(batch_invariant_ops, "ENABLE_JIT_DEEPGEMM", True),
            patch.object(
                batch_invariant_ops,
                "_matmul_persistent_triton",
                return_value=expected,
            ) as triton_matmul,
            patch.object(
                batch_invariant_ops, "_matmul_persistent_deepgemm"
            ) as deepgemm_matmul,
        ):
            actual = batch_invariant_ops.matmul_persistent(
                a, b, out_dtype=torch.float32
            )

        self.assertIs(actual, expected)
        triton_matmul.assert_called_once_with(
            a=a, b=b, bias=None, out_dtype=torch.float32
        )
        deepgemm_matmul.assert_not_called()

    def test_deepseek_deterministic_gate_requests_fp32(self):
        from sglang.srt.models.deepseek_v2 import MoEGate

        gate = MoEGate.__new__(MoEGate)
        torch.nn.Module.__init__(gate)
        gate.weight = torch.nn.Parameter(torch.randn((64, 257), dtype=torch.bfloat16))
        hidden_states = torch.randn((5, 257), dtype=torch.bfloat16)
        execution = SimpleNamespace(
            deterministic=SimpleNamespace(enable_deterministic_inference=True)
        )

        with (
            patch("sglang.srt.models.deepseek_v2.get_exec", return_value=execution),
            set_batch_invariant_mode(True),
        ):
            logits = gate(hidden_states)

        self.assertEqual(logits.dtype, torch.float32)
        self.assertGreater(
            torch.count_nonzero(logits != logits.bfloat16().float()).item(), 0
        )

    def test_medium_matrices(self):
        """Test batch invariance with medium matrix sizes"""
        test_cases = [
            ("Medium-1", 32, 128, 1024),
            ("Medium-2", 64, 512, 2048),
            ("Medium-3", 24, 192, 768),
        ]

        for name, M, K, N in test_cases:
            with self.subTest(name=name, M=M, K=K, N=N):
                for dtype in [torch.float32, torch.bfloat16]:
                    with self.subTest(dtype=dtype):
                        # Run with batch-invariant mode
                        with set_batch_invariant_mode(True):
                            difflist = self._run_multiple_iterations(
                                iters=5, M=M, K=K, N=N, dtype=dtype
                            )
                            self._assert_batch_invariant_results(difflist, dtype, name)

    def test_large_matrices(self):
        """Test batch invariance with large matrix sizes"""
        test_cases = [
            ("Large-1", 128, 1024, 4096),
            ("Large-2", 256, 2048, 8192),
            ("Large-3", 96, 768, 3072),
        ]

        for name, M, K, N in test_cases:
            with self.subTest(name=name, M=M, K=K, N=N):
                for dtype in [torch.float32, torch.bfloat16]:
                    with self.subTest(dtype=dtype):
                        # Run with batch-invariant mode
                        with set_batch_invariant_mode(True):
                            difflist = self._run_multiple_iterations(
                                iters=5, M=M, K=K, N=N, dtype=dtype
                            )
                            self._assert_batch_invariant_results(difflist, dtype, name)

    def _test_bmm_batch_invariance(self, B, M, K, N, dtype):
        """
        Test that BMM operations produce identical results for:
        - Method 1: BMM with subset of batches
        - Method 2: BMM with all batches, then slice
        """
        a = torch.linspace(-100, 100, B * M * K, dtype=dtype).reshape(B, M, K)
        b = torch.linspace(-100, 100, B * K * N, dtype=dtype).reshape(B, K, N)

        # Method 1: BMM with subset (first 2 batches)
        subset_size = min(2, B)
        out1 = torch.bmm(a[:subset_size], b[:subset_size])

        # Method 2: BMM with all batches, then slice
        out2_pre = torch.bmm(a, b)
        out2 = out2_pre[:subset_size]

        # Check if results are identical
        diff = (out1 - out2).abs().max()
        return diff.item()

    def _run_bmm_multiple_iterations(self, iters, B, M, K, N, dtype):
        """Run multiple BMM iterations and collect diff statistics"""
        difflist = []
        for _ in range(iters):
            diff = self._test_bmm_batch_invariance(B, M, K, N, dtype)
            difflist.append(diff)
        return difflist

    def test_bmm_small_matrices(self):
        """Test BMM batch invariance with small matrix sizes"""
        test_cases = [
            ("BMM-Small-1", 4, 8, 64, 128),
            ("BMM-Small-2", 8, 16, 128, 256),
            ("BMM-Small-3", 6, 4, 32, 64),
        ]

        for name, B, M, K, N in test_cases:
            with self.subTest(name=name, B=B, M=M, K=K, N=N):
                for dtype in [torch.float32, torch.bfloat16]:
                    with self.subTest(dtype=dtype):
                        # Run with batch-invariant mode
                        with set_batch_invariant_mode(True):
                            difflist = self._run_bmm_multiple_iterations(
                                iters=5, B=B, M=M, K=K, N=N, dtype=dtype
                            )
                            self._assert_batch_invariant_results(difflist, dtype, name)

    def test_bmm_medium_matrices(self):
        """Test BMM batch invariance with medium matrix sizes"""
        test_cases = [
            ("BMM-Medium-1", 8, 32, 128, 1024),
            ("BMM-Medium-2", 16, 64, 512, 2048),
            ("BMM-Medium-3", 12, 24, 192, 768),
        ]

        for name, B, M, K, N in test_cases:
            with self.subTest(name=name, B=B, M=M, K=K, N=N):
                for dtype in [torch.float32, torch.bfloat16]:
                    with self.subTest(dtype=dtype):
                        # Run with batch-invariant mode
                        with set_batch_invariant_mode(True):
                            difflist = self._run_bmm_multiple_iterations(
                                iters=5, B=B, M=M, K=K, N=N, dtype=dtype
                            )
                            self._assert_batch_invariant_results(difflist, dtype, name)

    def test_bmm_large_matrices(self):
        """Test BMM batch invariance with large matrix sizes"""
        test_cases = [
            ("BMM-Large-1", 16, 128, 1024, 4096),
            ("BMM-Large-2", 32, 256, 2048, 8192),
            ("BMM-Large-3", 24, 96, 768, 3072),
        ]

        for name, B, M, K, N in test_cases:
            with self.subTest(name=name, B=B, M=M, K=K, N=N):
                for dtype in [torch.float32, torch.bfloat16]:
                    with self.subTest(dtype=dtype):
                        # Run with batch-invariant mode
                        with set_batch_invariant_mode(True):
                            difflist = self._run_bmm_multiple_iterations(
                                iters=5, B=B, M=M, K=K, N=N, dtype=dtype
                            )
                            self._assert_batch_invariant_results(difflist, dtype, name)


if __name__ == "__main__":
    unittest.main()
