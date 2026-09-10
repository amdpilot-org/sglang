"""Backend opt-in flags for the ragged-verify graphs.

Runs in the GPU suite because importing the backend modules pulls GPU-only
wheels (sgl_kernel) at module scope, which fail to import on CPU runners.
"""

import unittest
from types import SimpleNamespace

from sglang.srt.environ import envs
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    ragged_verify_compact_graphs_enabled,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


class TestRaggedVerifyGraphCapability(CustomTestCase):
    def test_base_backend_defaults_false(self):
        from sglang.srt.layers.attention.base_attn_backend import AttentionBackend

        self.assertFalse(AttentionBackend.supports_ragged_verify_graph)

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

    def test_compact_capture_requires_backend_support(self):
        spec_algorithm = SimpleNamespace(supports_ragged_verify=lambda: True)
        with envs.SGLANG_RAGGED_VERIFY_MODE.override("compact"):
            self.assertTrue(
                ragged_verify_compact_graphs_enabled(
                    spec_algorithm,
                    SimpleNamespace(supports_ragged_verify_graph=True),
                )
            )
            self.assertFalse(
                ragged_verify_compact_graphs_enabled(
                    spec_algorithm,
                    SimpleNamespace(supports_ragged_verify_graph=False),
                )
            )


if __name__ == "__main__":
    unittest.main()
