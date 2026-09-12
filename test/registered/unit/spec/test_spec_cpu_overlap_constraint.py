import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.arg_groups.speculative_hook import handle_speculative_decoding
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _make_spec_args(device: str, algorithm: str = "EAGLE", **overrides) -> ServerArgs:
    # model_path="dummy" short-circuits ServerArgs.__post_init__; invoke the
    # speculative hook directly (same pattern as the unit/server_args tests).
    args = ServerArgs(model_path="dummy")
    args.speculative_algorithm = algorithm
    args.device = device
    # Fully specify the chain config so the hook doesn't auto-choose params.
    args.speculative_num_steps = 3
    args.speculative_eagle_topk = 1
    args.speculative_num_draft_tokens = 4
    args._model_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            architectures=["LlamaForCausalLM"],
            get_text_config=lambda: SimpleNamespace(),
        )
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


class TestSpecCPUOverlapConstraint(CustomTestCase):
    def test_cpu_eagle_forces_disable_overlap_schedule(self):
        args = _make_spec_args(device="cpu")
        self.assertFalse(resolution_result(args, "disable_overlap_schedule"))

        handle_speculative_decoding(args)

        self.assertTrue(resolution_result(args, "disable_overlap_schedule"))

    def test_cpu_eagle3_forces_disable_overlap_schedule(self):
        args = _make_spec_args(
            device="cpu",
            algorithm="EAGLE3",
            speculative_draft_model_path="dummy-draft",
        )

        with patch(
            "sglang.srt.utils.hf_transformers_utils.get_config",
            return_value=SimpleNamespace(architectures=["LlamaForCausalLM"]),
        ):
            handle_speculative_decoding(args)

        self.assertTrue(resolution_result(args, "disable_overlap_schedule"))

    def test_cpu_explicit_disable_overlap_is_preserved(self):
        args = _make_spec_args(device="cpu", disable_overlap_schedule=True)

        # Already disabled: the hook must not flip the flag, and (unlike the
        # forced-disable cases) must not warn about overriding it.
        with self.assertLogs(
            "sglang.srt.arg_groups.speculative_hook", "WARNING"
        ) as logs:
            handle_speculative_decoding(args)

        self.assertTrue(resolution_result(args, "disable_overlap_schedule"))
        self.assertFalse(
            any("Overlap schedule" in message for message in logs.output),
            f"hook warned about overriding an already-disabled overlap: {logs.output}",
        )

    def test_cuda_eagle_keeps_overlap_schedule(self):
        # Guard the constraint's scope: the hook must not touch non-CPU devices.
        args = _make_spec_args(device="cuda")

        handle_speculative_decoding(args)

        self.assertFalse(resolution_result(args, "disable_overlap_schedule"))

    def test_eagle3_requires_draft_model_for_ordinary_target(self):
        args = _make_spec_args(device="cuda", algorithm="EAGLE3")

        with self.assertRaisesRegex(
            ValueError, "EAGLE3.*--speculative-draft-model-path"
        ):
            handle_speculative_decoding(args)

    def test_eagle3_accepts_explicit_draft_model(self):
        args = _make_spec_args(
            device="cuda",
            algorithm="EAGLE3",
            speculative_draft_model_path="dummy-draft",
        )

        with patch(
            "sglang.srt.utils.hf_transformers_utils.get_config",
            return_value=SimpleNamespace(architectures=["LlamaForCausalLM"]),
        ):
            handle_speculative_decoding(args)

        self.assertEqual(
            resolution_result(args, "speculative_draft_model_path"), "dummy-draft"
        )

    def test_eagle3_uses_bundled_draft_model(self):
        args = _make_spec_args(device="cuda", algorithm="EAGLE3")
        args.model_path = "dummy-bundled-target"
        args.revision = "test-revision"
        args._model_config.hf_config.architectures = ["DeepseekV3ForCausalLM"]

        handle_speculative_decoding(args)

        self.assertEqual(
            resolution_result(args, "speculative_draft_model_path"),
            "dummy-bundled-target",
        )
        self.assertEqual(
            resolution_result(args, "speculative_draft_model_revision"),
            "test-revision",
        )

    def test_eagle_without_draft_model_remains_allowed(self):
        args = _make_spec_args(device="cuda", algorithm="EAGLE")

        handle_speculative_decoding(args)

        self.assertIsNone(resolution_result(args, "speculative_draft_model_path"))


if __name__ == "__main__":
    unittest.main()
