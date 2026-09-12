"""Regression tests for bounded incremental-detokenization windows."""

import unittest

from sglang.srt.managers.detokenizer_manager import DetokenizerManager
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

STRAY_CONTINUATION_BYTE = 0x80
CJK_BYTES = (0xE4, 0xB8, 0xAD)
TEXT_TOKEN = 256
MAX_EXPECTED_STALLED_TAIL = 64


class _ByteFallbackTokenizer:
    is_fast = True

    def batch_decode(self, ids_list, skip_special_tokens=True, **kwargs):
        return [self._decode(ids) for ids in ids_list]

    @staticmethod
    def _decode(ids):
        data = bytearray()
        for token_id in ids:
            data.append(token_id if token_id < 256 else ord("a"))
        return data.decode("utf-8", errors="replace")


class _Batch:
    def __init__(self, rid, decode_ids, read_offset):
        self.rids = [rid]
        self.finished_reasons = [None]
        self.decoded_texts = [""]
        self.decode_ids = [list(decode_ids)]
        self.read_offsets = [read_offset]
        self.skip_special_tokens = [True]
        self.spaces_between_special_tokens = [True]
        self.no_stop_trim = [False]


def _manager():
    manager = DetokenizerManager.__new__(DetokenizerManager)
    manager.tokenizer = _ByteFallbackTokenizer()
    manager.vocab_size = None
    manager.decode_status = {}
    manager.disable_tokenizer_batch_decode = False
    manager.is_tool_call_parser_gpt_oss = False
    return manager


def _stream(steps):
    manager = _manager()
    manager._decode_batch_token_id_output(
        _Batch("rid", [TEXT_TOKEN] + list(steps[0]), 1)
    )
    for step in steps[1:]:
        manager._decode_batch_token_id_output(_Batch("rid", step, 0))
    return manager.decode_status["rid"]


class TestIncrementalDetokenizationWindow(unittest.TestCase):
    def test_replacement_character_stream_has_bounded_window(self):
        state = _stream([[STRAY_CONTINUATION_BYTE] * 4] * 256)

        self.assertTrue(state.get_decoded_text())
        self.assertLess(
            len(state.decode_ids) - state.surr_offset,
            MAX_EXPECTED_STALLED_TAIL,
        )

    def test_oversized_event_does_not_leave_oversized_window(self):
        state = _stream([[STRAY_CONTINUATION_BYTE] * 4096] * 32)

        self.assertTrue(state.get_decoded_text())
        self.assertLess(
            len(state.decode_ids) - state.surr_offset,
            MAX_EXPECTED_STALLED_TAIL,
        )

    def test_split_multibyte_character_is_not_committed_early(self):
        state = _stream([[byte] for byte in CJK_BYTES])
        self.assertEqual(state.get_decoded_text(), "中")

    def test_clean_commits_reset_stall_tracking(self):
        steps = [[byte] for _ in range(32) for byte in CJK_BYTES]
        state = _stream(steps)
        self.assertEqual(state.get_decoded_text(), "中" * 32)


if __name__ == "__main__":
    unittest.main()
