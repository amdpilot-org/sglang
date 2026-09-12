import unittest
from unittest.mock import Mock, patch

from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@patch(
    "sglang.srt.managers.tokenizer_manager.get_server_return_hidden_states_mode",
    return_value=CaptureHiddenMode.NULL,
)
class TestAutoTruncateValidation(CustomTestCase):
    def _make_manager(self, *, allow_auto_truncate, reserved_tokens=0):
        manager = TokenizerManager.__new__(TokenizerManager)
        manager.context_len = 10
        manager.num_reserved_tokens = reserved_tokens
        manager.allow_auto_truncate = allow_auto_truncate
        manager.validate_total_tokens = True
        manager.is_generation = True
        manager._validate_token_ids_logprob = Mock()
        return manager

    @staticmethod
    def _make_request(input_ids, max_new_tokens):
        return GenerateReqInput(
            input_ids=input_ids,
            sampling_params={"max_new_tokens": max_new_tokens},
        )

    def test_truncates_completion_when_total_exceeds_context(self, _hidden_mode):
        manager = self._make_manager(allow_auto_truncate=True)
        request = self._make_request([1, 2, 3, 4, 5, 6], 8)

        manager._validate_one_request(request, request.input_ids)

        self.assertEqual(request.input_ids, [1, 2, 3, 4, 5, 6])
        self.assertEqual(request.sampling_params["max_new_tokens"], 4)

    def test_disabled_auto_truncate_preserves_error(self, _hidden_mode):
        manager = self._make_manager(allow_auto_truncate=False)
        request = self._make_request([1, 2, 3, 4, 5, 6], 8)

        with self.assertRaisesRegex(
            ValueError, "Requested token count exceeds.*14 tokens"
        ):
            manager._validate_one_request(request, request.input_ids)

        self.assertEqual(request.input_ids, [1, 2, 3, 4, 5, 6])
        self.assertEqual(request.sampling_params["max_new_tokens"], 8)

    def test_prompt_truncation_accounts_for_reserved_tokens(self, _hidden_mode):
        manager = self._make_manager(
            allow_auto_truncate=True,
            reserved_tokens=3,
        )
        request = self._make_request(list(range(12)), 5)

        manager._validate_one_request(request, request.input_ids)

        self.assertEqual(request.input_ids, list(range(7)))
        self.assertEqual(request.sampling_params["max_new_tokens"], 0)

    def test_reserved_tokens_reduce_completion_budget(self, _hidden_mode):
        manager = self._make_manager(
            allow_auto_truncate=True,
            reserved_tokens=3,
        )
        request = self._make_request([1, 2, 3, 4], 5)

        manager._validate_one_request(request, request.input_ids)

        self.assertEqual(request.input_ids, [1, 2, 3, 4])
        self.assertEqual(request.sampling_params["max_new_tokens"], 3)


if __name__ == "__main__":
    unittest.main()
