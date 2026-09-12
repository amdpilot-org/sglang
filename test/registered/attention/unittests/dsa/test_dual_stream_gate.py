import unittest

import numpy as np
import torch

from sglang.srt.layers.attention.dsa.dsa_indexer import (
    DUAL_STREAM_TOKEN_THRESHOLD,
    should_use_dsa_dual_stream,
)
from sglang.srt.utils import is_cuda, is_hip


class TestDSADualStreamGate(unittest.TestCase):
    def test_token_boundary(self):
        expected_threshold = 1024 if (is_cuda() or is_hip()) else 0
        self.assertEqual(DUAL_STREAM_TOKEN_THRESHOLD, expected_threshold)
        if expected_threshold == 0:
            self.skipTest("DSA dual streams require a CUDA-like platform")
        self.assertFalse(should_use_dsa_dual_stream(0, True, True))
        self.assertTrue(should_use_dsa_dual_stream(1, True, True))
        self.assertTrue(should_use_dsa_dual_stream(1024, True, True))
        self.assertFalse(should_use_dsa_dual_stream(1025, True, True))

    def test_stream_and_capture_prerequisites(self):
        self.assertFalse(should_use_dsa_dual_stream(1, False, True))
        self.assertFalse(should_use_dsa_dual_stream(1, True, False))

    @unittest.skipUnless(
        torch.cuda.is_available() and (is_cuda() or is_hip()),
        "requires a CUDA-like GPU",
    )
    def test_gpu_dual_stream_and_large_batch_fallback(self):
        alt_stream = torch.cuda.Stream()
        branch_results = {}

        for num_tokens in (1024, 1025):
            x_cpu = np.linspace(-1.0, 1.0, num_tokens * 8, dtype=np.float32).reshape(
                num_tokens, 8
            )
            x = torch.from_numpy(x_cpu).to("cuda")
            use_dual_stream = should_use_dsa_dual_stream(
                x.shape[0], alt_stream is not None, True
            )

            current_stream = torch.cuda.current_stream()
            if use_dual_stream:
                alt_stream.wait_stream(current_stream)
                with torch.cuda.stream(alt_stream):
                    actual = x.square().sum(dim=1)
                current_stream.wait_stream(alt_stream)
            else:
                actual = x.square().sum(dim=1)

            expected = np.square(x_cpu).sum(axis=1)
            np.testing.assert_allclose(
                actual.cpu().numpy(), expected, rtol=2e-6, atol=2e-6
            )
            branch_results[num_tokens] = use_dual_stream

        self.assertEqual(branch_results, {1024: True, 1025: False})


if __name__ == "__main__":
    unittest.main()
