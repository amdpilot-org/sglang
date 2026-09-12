import unittest
from types import SimpleNamespace

from sglang.srt.layers.attention.dsa.utils import cal_padded_tokens
from sglang.srt.layers.dp_attention import DpPaddingMode
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _forward_batch(global_num_tokens, padding_mode):
    return SimpleNamespace(
        attn_cp_metadata=None,
        global_num_tokens_cpu=global_num_tokens,
        dp_padding_mode=padding_mode,
        is_extend_in_batch=False,
        forward_mode=ForwardMode.DECODE,
    )


class TestDsaDpPadding(CustomTestCase):
    def _cal(self, counts, mode, *, attn_tp_size, attn_dp_rank):
        with get_parallel().override(
            attn_tp_size=attn_tp_size,
            attn_dp_rank=attn_dp_rank,
            attn_cp_size=1,
            attn_cp_rank=0,
        ):
            return cal_padded_tokens(_forward_batch(counts, mode))

    def test_max_len_aligns_each_dp_rank_before_selecting_max(self):
        for rank in (0, 1):
            with self.subTest(rank=rank):
                self.assertEqual(
                    self._cal(
                        [11, 1],
                        DpPaddingMode.MAX_LEN,
                        attn_tp_size=8,
                        attn_dp_rank=rank,
                    ),
                    16,
                )

    def test_sum_len_returns_aligned_local_rank_width(self):
        self.assertEqual(
            self._cal(
                [11, 1],
                DpPaddingMode.SUM_LEN,
                attn_tp_size=8,
                attn_dp_rank=0,
            ),
            16,
        )
        self.assertEqual(
            self._cal(
                [11, 1],
                DpPaddingMode.SUM_LEN,
                attn_tp_size=8,
                attn_dp_rank=1,
            ),
            8,
        )

    def test_aligned_counts_and_attention_tp_one_are_unchanged(self):
        self.assertEqual(
            self._cal(
                [16, 8],
                DpPaddingMode.SUM_LEN,
                attn_tp_size=8,
                attn_dp_rank=0,
            ),
            16,
        )
        self.assertEqual(
            self._cal(
                [11, 1],
                DpPaddingMode.SUM_LEN,
                attn_tp_size=1,
                attn_dp_rank=1,
            ),
            1,
        )


if __name__ == "__main__":
    unittest.main()
