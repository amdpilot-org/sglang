import ast
import unittest
from pathlib import Path
from unittest.mock import patch

from sglang.test.ascend.test_mmlu import TestMMLU as MMLUTestMixin
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

REPO_ROOT = Path(__file__).resolve().parents[3]
DEEPEP_TEST_DIR = (
    REPO_ROOT
    / "test/registered/npu/basic_function/parallel_strategy/expert_parallelism"
)
EXPECTED_THRESHOLDS = {
    "test_npu_deepep_auto_qwen3_480b.py": 0.61,
    "test_npu_deepep_auto_qwen3_next.py": 0.56,
    "test_npu_deepep_low_latency_deepseek_v3_2_w8a8.py": 0.85,
    "test_npu_deepep_low_latency_qwen3_480b.py": 0.61,
    "test_npu_deepep_low_latency_qwen3_next.py": 0.56,
}


class TestMMLUConfig(unittest.TestCase):
    def test_deepep_tests_use_consumed_threshold_name(self):
        for filename, expected in EXPECTED_THRESHOLDS.items():
            with self.subTest(filename=filename):
                tree = ast.parse((DEEPEP_TEST_DIR / filename).read_text())
                assignments = {
                    target.id: ast.literal_eval(node.value)
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Assign)
                    for target in node.targets
                    if isinstance(target, ast.Name)
                    and target.id in {"accuracy_mmlu", "accuracy_mmlu_threshold"}
                }
                self.assertEqual(assignments.get("accuracy_mmlu"), expected)
                self.assertNotIn("accuracy_mmlu_threshold", assignments)

    def _run_mmlu(self, score, threshold=None):
        class MMLUCase(MMLUTestMixin, unittest.TestCase):
            base_url = "http://unused"
            model = "fixture"
            other_args = []

        case = MMLUCase("test_mmlu")
        if threshold is not None:
            case.accuracy_mmlu = threshold
        with (
            patch(
                "sglang.test.ascend.test_mmlu.run_eval",
                return_value={"score": score},
            ),
            patch("sglang.test.ascend.test_mmlu.write_results_to_github_step_summary"),
        ):
            return case.run()

    def test_configured_threshold_rejects_lower_score(self):
        result = self._run_mmlu(score=0.60, threshold=0.61)
        self.assertFalse(result.wasSuccessful())

    def test_configured_threshold_accepts_higher_score(self):
        result = self._run_mmlu(score=0.62, threshold=0.61)
        self.assertTrue(result.wasSuccessful())

    def test_default_threshold_accepts_positive_score(self):
        result = self._run_mmlu(score=0.01)
        self.assertTrue(result.wasSuccessful())


if __name__ == "__main__":
    unittest.main()
