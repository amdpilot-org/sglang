"""Backend opt-in flags for the ragged-verify graphs.

Runs in the GPU suite because importing the backend modules pulls GPU-only
wheels (sgl_kernel) at module scope, which fail to import on CPU runners.
"""

import unittest
from unittest.mock import Mock, patch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


class TestRaggedVerifyGraphCapability(CustomTestCase):
    def test_base_backend_defaults_false(self):
        from sglang.srt.layers.attention.base_attn_backend import AttentionBackend

        self.assertFalse(AttentionBackend.supports_ragged_verify_graph)

    def test_triton_hybrid_requires_both_children_to_support_ragged_graphs(self):
        from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
            HybridLinearAttnBackend,
        )
        from sglang.srt.layers.attention.triton_backend import TritonAttnBackend

        self.assertFalse(TritonAttnBackend.supports_ragged_verify_graph)

        hybrid = HybridLinearAttnBackend.__new__(HybridLinearAttnBackend)
        hybrid.full_attn_backend = Mock(supports_ragged_verify_graph=False)
        hybrid.linear_attn_backend = Mock(supports_ragged_verify_graph=True)
        self.assertFalse(hybrid.supports_ragged_verify_graph)

        hybrid.full_attn_backend.supports_ragged_verify_graph = True
        self.assertTrue(hybrid.supports_ragged_verify_graph)

    def test_ragged_implementing_backends_declare_the_flag(self):
        """Every backend with a ragged-verify metadata path must opt in; a
        dropped flag silently disables ragged graphs for that backend (the
        runner falls back to eager with no other test going red)."""
        from sglang.srt.layers.attention.deepseek_v4_backend import (
            DeepseekV4AttnBackend,
        )
        from sglang.srt.layers.attention.flashattention_backend import (
            FlashAttentionBackend,
        )
        from sglang.srt.layers.attention.trtllm_mha_backend import TRTLLMHAAttnBackend

        for backend in (
            TRTLLMHAAttnBackend,
            DeepseekV4AttnBackend,
            FlashAttentionBackend,
        ):
            with self.subTest(backend=backend.__name__):
                self.assertTrue(backend.supports_ragged_verify_graph)

    def test_compact_capture_skips_unsupported_backend_and_replay(self):
        from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
            DecodeCudaGraphRunner,
        )

        runner = DecodeCudaGraphRunner.__new__(DecodeCudaGraphRunner)
        runner.ragged_verify_mode = True
        runner.attn_backend = Mock(supports_ragged_verify_graph=False)
        runner.warmup = Mock(side_effect=AssertionError("must not warm up"))

        with patch(
            "sglang.srt.model_executor.runner.decode_cuda_graph_runner.logger.warning"
        ) as warning:
            runner.capture()

        runner.warmup.assert_not_called()
        warning.assert_called_once()
        self.assertIn("eager path", warning.call_args.args[0])
        self.assertFalse(
            runner._can_run_ragged_verify_graph(Mock(), Mock()),
            "unsupported compact verification must remain on the eager path",
        )

    def test_supported_compact_capture_continues(self):
        from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
            DecodeCudaGraphRunner,
        )

        runner = DecodeCudaGraphRunner.__new__(DecodeCudaGraphRunner)
        runner.ragged_verify_mode = True
        runner.attn_backend = Mock(supports_ragged_verify_graph=True)
        runner.warmup = Mock(side_effect=RuntimeError("capture continued"))

        with self.assertRaisesRegex(RuntimeError, "capture continued"):
            runner.capture()

    def test_non_compact_capture_ignores_ragged_capability(self):
        from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
            DecodeCudaGraphRunner,
        )

        runner = DecodeCudaGraphRunner.__new__(DecodeCudaGraphRunner)
        runner.ragged_verify_mode = False
        runner.attn_backend = Mock(supports_ragged_verify_graph=False)
        runner.warmup = Mock(side_effect=RuntimeError("capture continued"))

        with self.assertRaisesRegex(RuntimeError, "capture continued"):
            runner.capture()


if __name__ == "__main__":
    unittest.main()
