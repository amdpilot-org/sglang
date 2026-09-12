import unittest

from sglang.srt.utils.request_logger import (
    _dataclass_to_string_truncated,
    _transform_data_for_logging,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestRequestLoggerTruncation(unittest.TestCase):
    def test_json_elided_list_still_truncates_elements(self):
        result = _transform_data_for_logging(["abcdefghij"] * 5, max_length=4)

        self.assertEqual(result, ["ab...ij", "ab...ij", "...", "ab...ij", "ab...ij"])

    def test_text_list_truncates_elements(self):
        result = _dataclass_to_string_truncated(
            ["abcdefghij", "klmnopqrst"], max_length=4
        )

        self.assertEqual(result, "['ab' ... 'ij', 'kl' ... 'st']")

    def test_list_at_length_boundary_is_not_elided(self):
        result = _transform_data_for_logging(
            ["abcdefghij", {"value": "klmnopqrst"}], max_length=2
        )

        self.assertEqual(result, ["a...j", {"value": "k...t"}])

    def test_text_elided_tuple_preserves_tuple_format(self):
        result = _dataclass_to_string_truncated(
            ("abcdefghij", "middle", "klmnopqrst"), max_length=2
        )

        self.assertEqual(result, "('a' ... 'j',) ... ('k' ... 't',)")

    def test_json_max_length_one_does_not_retain_full_tail(self):
        result = _transform_data_for_logging(["x" * 10_000] * 2, max_length=1)

        self.assertEqual(result, ["..."])

    def test_text_max_length_one_does_not_retain_full_tail(self):
        result = _dataclass_to_string_truncated(["x" * 10_000] * 2, max_length=1)

        self.assertEqual(result, "[] ... []")

    def test_string_max_length_one_does_not_retain_full_tail(self):
        result = _transform_data_for_logging("x" * 10_000, max_length=1)

        self.assertEqual(result, "...")


if __name__ == "__main__":
    unittest.main()
