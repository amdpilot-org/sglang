import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.dsa.utils import pad_dsa_cache_seqlens
from sglang.srt.layers.dp_attention import DpPaddingMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDsaDpSpecPadding(unittest.TestCase):
    def _forward_batch(self, global_tokens, mode, *, cp_metadata=None):
        return SimpleNamespace(
            global_num_tokens_cpu=global_tokens,
            dp_padding_mode=mode,
            is_extend_in_batch=False,
            attn_cp_metadata=cp_metadata,
        )

    @patch("sglang.srt.layers.attention.dsa.utils.can_dsa_prefill_cp_interleave")
    @patch("sglang.srt.layers.cp.utils.is_cp_active")
    @patch("sglang.srt.layers.attention.dsa.utils.get_parallel")
    def test_spec_decode_uses_final_attention_tp_padded_size(
        self, get_parallel, is_cp_active, can_cp_interleave
    ):
        """FlashMLA num_splits must be one longer than the padded query batch."""
        get_parallel.return_value = SimpleNamespace(
            attn_cp_size=1, attn_dp_rank=0, attn_tp_size=4
        )
        is_cp_active.return_value = False
        can_cp_interleave.return_value = False

        # Eagle pre-plans metadata before prepare_mlp_sync_batch aligns this DP
        # rank from 9 to 12. DSA must anticipate the eventual 12 query rows.
        batch = self._forward_batch([9, 8], DpPaddingMode.SUM_LEN)
        padded = pad_dsa_cache_seqlens(batch, torch.arange(9, dtype=torch.int32))

        self.assertEqual(padded.shape, (12,))
        self.assertEqual(padded.tolist(), list(range(9)) + [0, 0, 0])
        self.assertEqual(padded.numel() + 1, 13)

    @patch("sglang.srt.layers.attention.dsa.utils.can_dsa_prefill_cp_interleave")
    @patch("sglang.srt.layers.cp.utils.is_cp_active")
    @patch("sglang.srt.layers.attention.dsa.utils.get_parallel")
    def test_sum_len_selects_current_dp_rank(
        self, get_parallel, is_cp_active, can_cp_interleave
    ):
        get_parallel.return_value = SimpleNamespace(
            attn_cp_size=1, attn_dp_rank=1, attn_tp_size=4
        )
        is_cp_active.return_value = False
        can_cp_interleave.return_value = False

        batch = self._forward_batch([9, 6], DpPaddingMode.SUM_LEN)
        padded = pad_dsa_cache_seqlens(batch, torch.arange(6, dtype=torch.int32))

        self.assertEqual(padded.shape, (8,))

    @patch("sglang.srt.layers.attention.dsa.utils.can_dsa_prefill_cp_interleave")
    @patch("sglang.srt.layers.cp.utils.is_cp_active")
    @patch("sglang.srt.layers.attention.dsa.utils.get_parallel")
    def test_max_len_uses_shared_padded_size(
        self, get_parallel, is_cp_active, can_cp_interleave
    ):
        get_parallel.return_value = SimpleNamespace(
            attn_cp_size=1, attn_dp_rank=1, attn_tp_size=4
        )
        is_cp_active.return_value = False
        can_cp_interleave.return_value = False

        batch = self._forward_batch([9, 6], DpPaddingMode.MAX_LEN)
        padded = pad_dsa_cache_seqlens(batch, torch.arange(8, dtype=torch.int32))

        self.assertEqual(padded.shape, (12,))

    @patch("sglang.srt.layers.attention.dsa.utils.get_parallel")
    def test_without_dp_or_cp_padding_preserves_metadata(self, get_parallel):
        get_parallel.return_value = SimpleNamespace(attn_cp_size=1, attn_tp_size=4)
        batch = self._forward_batch(None, None)
        cache_seqlens = torch.arange(5, dtype=torch.int32)

        self.assertIs(pad_dsa_cache_seqlens(batch, cache_seqlens), cache_seqlens)


if __name__ == "__main__":
    unittest.main()
