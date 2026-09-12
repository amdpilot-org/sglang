from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups.deepseek_v4_hook import apply_deepseek_v4_defaults
from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.arg_groups.speculative_hook import handle_speculative_decoding
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestDeepseekV4SpeculativeAlgorithm(CustomTestCase):
    def _args(self, algorithm: str, topk: int = 1) -> ServerArgs:
        args = ServerArgs(
            model_path="dummy",
            device="cuda",
            speculative_algorithm=algorithm,
            speculative_num_steps=2,
            speculative_eagle_topk=topk,
            speculative_num_draft_tokens=3,
        )
        args._model_config = SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["DeepseekV4ForCausalLM"])
        )
        args._resolved_overrides = []
        return args

    def _apply_defaults(self, args: ServerArgs) -> None:
        with patch(
            "sglang.srt.arg_groups.deepseek_v4_hook.get_platform",
            return_value=SimpleNamespace(is_hip=False),
        ):
            apply_deepseek_v4_defaults(args, "DeepseekV4ForCausalLM")

    def test_nextn_is_accepted_then_canonicalized_to_eagle(self):
        args = self._args("NEXTN")

        self._apply_defaults(args)
        handle_speculative_decoding(args)

        self.assertEqual(resolution_result(args, "speculative_algorithm"), "EAGLE")

    def test_nextn_requires_single_eagle_candidate(self):
        with self.assertRaisesRegex(AssertionError, "topk == 1"):
            self._apply_defaults(self._args("NEXTN", topk=2))

    def test_existing_supported_algorithms_remain_accepted(self):
        for algorithm in ("EAGLE", "DSPARK"):
            with self.subTest(algorithm=algorithm):
                self._apply_defaults(self._args(algorithm))

    def test_unrelated_speculative_algorithm_remains_rejected(self):
        with self.assertRaisesRegex(
            AssertionError, "Only EAGLE, NEXTN and DSPARK speculative algorithms"
        ):
            self._apply_defaults(self._args("NGRAM"))
