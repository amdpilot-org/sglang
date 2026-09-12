import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups.deepseek_v4_hook import apply_deepseek_v4_defaults
from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.arg_groups.speculative_hook import (
    _handle_dspark,
    _handle_eagle_family,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDeepSeekV4SpeculativeMaxRunningRequests(CustomTestCase):
    def _args(self, max_running_requests=None, algorithm="EAGLE"):
        args = ServerArgs(
            model_path="dummy",
            device="cuda",
            speculative_algorithm=algorithm,
            speculative_draft_model_path=(
                "dummy-draft" if algorithm == "DSPARK" else None
            ),
            speculative_num_steps=3,
            speculative_eagle_topk=1,
            speculative_num_draft_tokens=4,
            max_running_requests=max_running_requests,
        )
        args._raw_input = {
            "max_running_requests": max_running_requests,
        }
        args._resolved_overrides = []
        args._model_config = SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["DeepseekV4ForCausalLM"])
        )
        return args

    def _apply_model_then_speculative_defaults(self, args):
        platform = SimpleNamespace(is_hip=False)
        with patch(
            "sglang.srt.arg_groups.deepseek_v4_hook.get_platform",
            return_value=platform,
        ):
            apply_deepseek_v4_defaults(args, "DeepseekV4ForCausalLM")
        if args.speculative_algorithm == "DSPARK":
            with patch(
                "sglang.srt.speculative.dspark_components.dspark_config.read_draft_checkpoint_config",
                return_value=None,
            ):
                _handle_dspark(args)
        else:
            _handle_eagle_family(args)

    def test_speculative_default_overrides_deepseek_v4_model_default(self):
        for algorithm in ("EAGLE", "DSPARK"):
            with self.subTest(algorithm=algorithm):
                args = self._args(algorithm=algorithm)

                self._apply_model_then_speculative_defaults(args)

                self.assertEqual(48, resolution_result(args, "max_running_requests"))

    def test_explicit_value_keeps_precedence(self):
        args = self._args(max_running_requests=96)

        self._apply_model_then_speculative_defaults(args)

        self.assertEqual(96, resolution_result(args, "max_running_requests"))

    def test_non_speculative_deepseek_v4_default_remains_256(self):
        args = self._args()
        args.speculative_algorithm = None
        args._raw_input["speculative_algorithm"] = None
        platform = SimpleNamespace(is_hip=False)

        with patch(
            "sglang.srt.arg_groups.deepseek_v4_hook.get_platform",
            return_value=platform,
        ):
            apply_deepseek_v4_defaults(args, "DeepseekV4ForCausalLM")

        self.assertEqual(256, resolution_result(args, "max_running_requests"))


if __name__ == "__main__":
    unittest.main()
