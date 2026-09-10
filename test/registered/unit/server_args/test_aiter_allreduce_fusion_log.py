import logging
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups.model_hook import handle_model_specific_adjustments
from sglang.srt.arg_groups.model_override_base import resolved_view
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    CudaGraphConfig,
    PhaseConfig,
)
from sglang.srt.runtime_context import override_platform
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestAiterAllreduceFusionLog(CustomTestCase):
    def _resolve(self, architecture, *, explicit=False, declared=False):
        server_args = ServerArgs(model_path="dummy")
        server_args._model_config = SimpleNamespace(
            hf_config=SimpleNamespace(architectures=[architecture])
        )
        server_args._resolved_overrides = []
        server_args.cuda_graph_config = CudaGraphConfig(
            decode=PhaseConfig(backend=Backend.FULL, max_bs=512),
            prefill=PhaseConfig(backend=Backend.DISABLED),
        )
        if explicit:
            server_args.enable_aiter_allreduce_fusion = True
        if declared:
            server_args._resolved_overrides.append(
                ("test_override", {"enable_aiter_allreduce_fusion": True})
            )
        if architecture == "GptOssForCausalLM":
            server_args.attention_backend = "aiter"

        logger = logging.getLogger("sglang.srt.arg_groups.model_hook")
        messages = []
        handler = logging.Handler()
        handler.emit = lambda record: messages.append(record.getMessage())
        logger.addHandler(handler)
        previous_level = logger.level
        logger.setLevel(logging.INFO)
        try:
            with override_platform(is_hip=True):
                with patch(
                    "sglang.srt.configs.model_config.is_deepseek_dsa",
                    return_value=False,
                ):
                    with patch(
                        "sglang.srt.arg_groups.model_hook.run_post_process_pass"
                    ):
                        handle_model_specific_adjustments(server_args)
        finally:
            logger.removeHandler(handler)
            logger.setLevel(previous_level)

        enabled_messages = [
            message
            for message in messages
            if "Enable Aiter AllReduce Fusion" in message
        ]
        return server_args, enabled_messages

    def test_deepseek_and_glm_log_only_when_resolved_enabled(self):
        for architecture in (
            "DeepseekV3ForCausalLM",
            "GlmMoeDsaForCausalLM",
            "GptOssForCausalLM",
        ):
            with self.subTest(architecture=architecture, explicit=False):
                server_args, messages = self._resolve(architecture)
                self.assertFalse(server_args.enable_aiter_allreduce_fusion)
                self.assertFalse(
                    resolved_view(server_args).enable_aiter_allreduce_fusion
                )
                self.assertEqual(messages, [])

            with self.subTest(architecture=architecture, explicit=True):
                server_args, messages = self._resolve(architecture, explicit=True)
                self.assertTrue(server_args.enable_aiter_allreduce_fusion)
                self.assertTrue(
                    resolved_view(server_args).enable_aiter_allreduce_fusion
                )
                self.assertEqual(len(messages), 1)

    def test_deepseek_logs_model_override_resolution(self):
        server_args, messages = self._resolve(
            "DeepseekV3ForCausalLM", declared=True
        )
        self.assertFalse(server_args.enable_aiter_allreduce_fusion)
        self.assertTrue(resolved_view(server_args).enable_aiter_allreduce_fusion)
        self.assertEqual(len(messages), 1)


if __name__ == "__main__":
    unittest.main()
