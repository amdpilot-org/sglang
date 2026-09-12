import unittest

from sglang.srt.arg_groups.speculative_hook import _validate_native_mtp_algorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestGlm5NextSpecAlgorithm(CustomTestCase):
    def test_rejects_eagle3_with_bundled_nextn_head(self):
        with self.assertRaisesRegex(ValueError, "trained for EAGLE/NEXTN.*not EAGLE3"):
            _validate_native_mtp_algorithm(
                model_arch="Glm5NextForConditionalGeneration",
                speculative_algorithm="EAGLE3",
                model_path="zai-org/GLM-5.3-Flash",
                speculative_draft_model_path="zai-org/GLM-5.3-Flash",
            )

    def test_rejects_eagle3_with_implicit_bundled_nextn_head(self):
        with self.assertRaisesRegex(ValueError, "Use --speculative-algorithm NEXTN"):
            _validate_native_mtp_algorithm(
                model_arch="Glm5NextForConditionalGeneration",
                speculative_algorithm="EAGLE3",
                model_path="zai-org/GLM-5.3-Flash",
                speculative_draft_model_path=None,
            )

    def test_accepts_eagle_with_bundled_nextn_head(self):
        _validate_native_mtp_algorithm(
            model_arch="Glm5NextForConditionalGeneration",
            speculative_algorithm="EAGLE",
            model_path="zai-org/GLM-5.3-Flash",
            speculative_draft_model_path="zai-org/GLM-5.3-Flash",
        )

    def test_accepts_distinct_eagle3_draft(self):
        _validate_native_mtp_algorithm(
            model_arch="Glm5NextForConditionalGeneration",
            speculative_algorithm="EAGLE3",
            model_path="zai-org/GLM-5.3-Flash",
            speculative_draft_model_path="org/trained-glm5-eagle3-draft",
        )

    def test_accepts_eagle3_for_other_target_architecture(self):
        _validate_native_mtp_algorithm(
            model_arch="LlamaForCausalLM",
            speculative_algorithm="EAGLE3",
            model_path="org/llama",
            speculative_draft_model_path="org/llama",
        )


if __name__ == "__main__":
    unittest.main()
