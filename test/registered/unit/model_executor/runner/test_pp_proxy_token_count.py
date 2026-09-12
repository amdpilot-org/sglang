import unittest

from sglang.srt.model_executor.runner.base_runner import get_pp_proxy_num_tokens


class TestPPProxyTokenCount(unittest.TestCase):
    def test_scattered_pp_boundary_uses_rank_local_rows(self):
        self.assertEqual(
            get_pp_proxy_num_tokens(32, require_attn_tp_gather=True, attn_tp_size=4),
            8,
        )

    def test_full_pp_boundary_keeps_global_rows(self):
        self.assertEqual(
            get_pp_proxy_num_tokens(32, require_attn_tp_gather=False, attn_tp_size=4),
            32,
        )

    def test_scattered_pp_boundary_rejects_non_divisible_rows(self):
        with self.assertRaisesRegex(ValueError, "divisible"):
            get_pp_proxy_num_tokens(10, require_attn_tp_gather=True, attn_tp_size=4)

    def test_scattered_pp_boundary_rejects_invalid_tp_size(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            get_pp_proxy_num_tokens(32, require_attn_tp_gather=True, attn_tp_size=0)


if __name__ == "__main__":
    unittest.main()
