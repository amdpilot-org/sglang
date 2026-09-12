"""Regression coverage for streaming detokenizer recovery offsets."""

import unittest
from types import SimpleNamespace

from sglang.srt.managers.detokenizer_manager import DetokenizerManager
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _TableTokenizer:
    def __init__(self, table):
        self.table = table

    def decode(self, ids, skip_special_tokens=True, spaces_between_special_tokens=True):
        return self.table[tuple(ids)]


def _recv(rows):
    """Build the fields consumed by the real manager decoding method."""
    return SimpleNamespace(
        rids=[row[0] for row in rows],
        decoded_texts=[row[1] for row in rows],
        decode_ids=[row[2] for row in rows],
        read_offsets=[row[3] for row in rows],
        finished_reasons=[row[4] for row in rows],
        no_stop_trim=[False] * len(rows),
        skip_special_tokens=[True] * len(rows),
        spaces_between_special_tokens=[True] * len(rows),
    )


class TestDetokenizerStreamOffset(CustomTestCase):
    def _manager(self, table):
        manager = DetokenizerManager.__new__(DetokenizerManager)
        manager.decode_status = {}
        manager.disable_tokenizer_batch_decode = True
        manager.tokenizer = _TableTokenizer(table)
        manager.vocab_size = None
        manager.is_tool_call_parser_gpt_oss = False
        return manager

    def test_repeated_recovery_does_not_duplicate_cjk(self):
        manager = self._manager(
            {
                (): "",
                (0,): "A 世�",
                (0, 1): "A 世a�",
                (0, 1, 2): "A 世ab�",
                (0, 1, 2, 3): "A 世abc�",
                (0, 1, 2, 3, 4): "A 世abcd",
            }
        )

        chunks = []
        for token in range(5):
            chunks.extend(
                manager._decode_batch_token_id_output(
                    _recv([("cjk", "", [token], 0, None)])
                )
            )

        # This expectation is deliberately independent of manager offsets: the
        # client concatenates the chunks and must receive the decoded fixture once.
        self.assertEqual(chunks, ["A 世", "", "", "", "abcd"])
        self.assertEqual("".join(chunks), "A 世abcd")

    def test_recovery_is_isolated_per_request_in_a_batch(self):
        manager = self._manager(
            {
                (): "",
                (0,): "A 世�",
                (0, 1): "A 世a�",
                (0, 1, 2): "A 世ab",
                (10,): "x",
                (10, 11): "xy",
                (11,): "y",
                (11, 12): "yz",
                (12,): "z",
            }
        )
        cjk_chunks, clean_chunks = [], []
        for cjk_token, clean_token in zip(range(3), range(10, 13)):
            output = manager._decode_batch_token_id_output(
                _recv(
                    [
                        ("cjk", "", [cjk_token], 0, None),
                        ("clean", "", [clean_token], 0, None),
                    ]
                )
            )
            cjk_chunks.append(output[0])
            clean_chunks.append(output[1])

        self.assertEqual("".join(cjk_chunks), "A 世ab")
        self.assertEqual(clean_chunks, ["x", "y", "z"])
        self.assertEqual("".join(clean_chunks), "xyz")

    def test_finished_step_after_recovery_emits_only_unsent_tail(self):
        manager = self._manager(
            {
                (): "",
                (0,): "A 世�",
                (0, 1): "A 世a�",
                (0, 1, 2): "A 世ab",
            }
        )
        chunks = []
        for token, reason in [(0, None), (1, None), (2, {"type": "length"})]:
            chunks.extend(
                manager._decode_batch_token_id_output(
                    _recv([("finish", "", [token], 0, reason)])
                )
            )

        self.assertEqual(chunks, ["A 世", "", "ab"])
        self.assertEqual("".join(chunks), "A 世ab")
        self.assertNotIn("finish", manager.decode_status)

    def test_non_streaming_materializes_clean_text_once(self):
        manager = self._manager({(): "", (20,): "A 世ab"})
        output = manager._decode_batch_token_id_output(
            _recv([("non-stream", "", [20], 0, {"type": "length"})])
        )

        self.assertEqual(output, ["A 世ab"])
        self.assertNotIn("non-stream", manager.decode_status)


if __name__ == "__main__":
    unittest.main()
