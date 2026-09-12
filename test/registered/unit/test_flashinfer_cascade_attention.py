import math
import types
import unittest
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.attention import flashinfer_backend as backend


class TestCascadeSelection(unittest.TestCase):
    def test_common_prefix_stops_at_first_mismatch(self):
        rows = torch.tensor(
            [[10, 11, 12, 13, 14], [10, 11, 99, 13, 14], [10, 11, 12, 13, 14]]
        )
        self.assertEqual(backend.common_prefix_length(rows), 2)

    def test_common_prefix_rejects_non_matrix(self):
        with self.assertRaisesRegex(ValueError, "2-D token table"):
            backend.common_prefix_length(torch.tensor([1, 2, 3]))

    def test_profitability_boundaries(self):
        cases = [
            ((3, 900, 1000), False),
            ((4, 511, 600), False),
            ((4, 512, 1025), False),
            ((4, 800, 1000), True),
            ((15, 600, 1000), False),
            ((16, 600, 1000), True),
            ((32, 512, 0), False),
        ]
        for args, expected in cases:
            with self.subTest(args=args):
                self.assertEqual(backend.should_use_cascade_attention(*args), expected)


class TestCascadeForward(unittest.TestCase):
    def test_index_updater_splits_shared_and_unique_page_tables(self):
        token_table = torch.tensor(
            [
                [10, 11, 12, 20, 21, 22],
                [10, 11, 12, 30, 31, 32],
                [10, 11, 12, 40, 41, 42],
                [10, 11, 12, 50, 51, 52],
            ],
            dtype=torch.int32,
        )

        class Translator:
            def fill_packed_read_stream(
                self,
                *,
                req_pool_indices,
                seq_lens,
                indptr,
                total_tokens,
                out,
                kv_start_idx,
                sliding_window,
            ):
                self.assertFalse(sliding_window)
                offset = 0
                for request_idx, (row, length) in enumerate(
                    zip(req_pool_indices.tolist(), seq_lens.tolist())
                ):
                    start = (
                        0 if kv_start_idx is None else int(kv_start_idx[request_idx])
                    )
                    out[offset : offset + length] = token_table[
                        row, start : start + length
                    ]
                    offset += length

            assertFalse = staticmethod(unittest.TestCase().assertFalse)

        updater = backend.FlashInferIndicesUpdaterCascadeDecode.__new__(
            backend.FlashInferIndicesUpdaterCascadeDecode
        )
        updater.num_qo_heads = 8
        updater.num_kv_heads = 2
        updater.head_dim = 64
        updater.data_type = torch.float16
        updater.q_data_type = torch.float16
        updater.attn_backend = types.SimpleNamespace(
            kv_index_translator=Translator()
        )
        wrappers = [Mock(), Mock()]
        updater.update(
            torch.arange(4, dtype=torch.int32),
            torch.tensor([6, 5, 6, 4], dtype=torch.int32),
            torch.tensor([6, 5, 6, 4], dtype=torch.int32),
            3,
            wrappers,
        )

        shared_args = wrappers[0].begin_forward.call_args.args
        unique_args = wrappers[1].begin_forward.call_args.args
        torch.testing.assert_close(
            shared_args[0], torch.tensor([0, 4], dtype=torch.int32)
        )
        torch.testing.assert_close(
            shared_args[1], torch.tensor([0, 3], dtype=torch.int32)
        )
        torch.testing.assert_close(
            shared_args[2], torch.tensor([10, 11, 12], dtype=torch.int32)
        )
        torch.testing.assert_close(
            unique_args[0], torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32)
        )
        torch.testing.assert_close(
            unique_args[1], torch.tensor([0, 3, 5, 8, 9], dtype=torch.int32)
        )
        torch.testing.assert_close(
            unique_args[2],
            torch.tensor([20, 21, 22, 30, 31, 40, 41, 42, 50], dtype=torch.int32),
        )

    def test_forwards_layer_scales_and_merges_states(self):
        shared = Mock()
        unique = Mock()
        shared.forward_return_lse.return_value = (
            torch.full((2, 3, 4), 1.0),
            torch.full((2, 3), 2.0),
        )
        unique.forward_return_lse.return_value = (
            torch.full((2, 3, 4), 3.0),
            torch.full((2, 3), 4.0),
        )
        attn = backend.FlashInferAttnBackend.__new__(backend.FlashInferAttnBackend)
        attn.forward_metadata = backend.DecodeMetadata(
            decode_wrappers=[Mock()], cascade_wrappers=[shared, unique]
        )
        attn.num_wrappers = 1
        attn.decode_uses_dequant_workspace = False
        attn.token_to_kv_pool = types.SimpleNamespace(
            get_kv_buffer=Mock(return_value="kv-cache")
        )
        layer = types.SimpleNamespace(
            is_cross_attention=False,
            tp_q_head_num=3,
            head_dim=4,
            layer_id=7,
            scaling=0.125,
            logit_cap=30.0,
            k_scale_float=0.5,
            v_scale_float=0.25,
        )
        batch = types.SimpleNamespace(out_cache_loc=None, encoder_out_cache_loc=None)
        merged = torch.full((2, 3, 4), 7.0)

        with patch.object(
            backend, "_safe_merge_state", return_value=(merged, None), create=True
        ):
            output = attn.forward_decode(
                torch.zeros(2, 12), None, None, layer, batch, save_kv_cache=False
            )

        self.assertTrue(torch.equal(output, merged.view(2, 12)))
        expected_kwargs = dict(
            causal=False,
            sm_scale=0.125,
            logits_soft_cap=30.0,
            k_scale=0.5,
            v_scale=0.25,
        )
        shared.forward_return_lse.assert_called_once_with(
            unittest.mock.ANY, "kv-cache", **expected_kwargs
        )
        unique.forward_return_lse.assert_called_once_with(
            unittest.mock.ANY, "kv-cache", **expected_kwargs
        )


@unittest.skipUnless(torch.cuda.is_available(), "requires a GPU")
class TestCascadeNumerics(unittest.TestCase):
    def test_split_softmax_matches_independent_full_attention(self):
        torch.manual_seed(17)
        device = "cuda"
        batch, heads, dim = 5, 4, 32
        shared_len, unique_len = 23, 11
        q = torch.randn(batch, heads, dim, device=device, dtype=torch.float64)
        shared_k = torch.randn(
            shared_len, heads, dim, device=device, dtype=torch.float64
        )
        shared_v = torch.randn(
            shared_len, heads, dim, device=device, dtype=torch.float64
        )
        unique_k = torch.randn(
            batch, unique_len, heads, dim, device=device, dtype=torch.float64
        )
        unique_v = torch.randn_like(unique_k)
        scale = 1 / math.sqrt(dim)

        def attention(k, v):
            scores = torch.einsum("bhd,bthd->bht", q, k) * scale
            lse = torch.logsumexp(scores, dim=-1)
            out = torch.einsum("bht,bthd->bhd", scores.softmax(dim=-1), v)
            return out, lse

        expanded_shared_k = shared_k.unsqueeze(0).expand(batch, -1, -1, -1)
        expanded_shared_v = shared_v.unsqueeze(0).expand(batch, -1, -1, -1)
        shared_out, shared_lse = attention(expanded_shared_k, expanded_shared_v)
        unique_out, unique_lse = attention(unique_k, unique_v)
        normalizer = torch.logaddexp(shared_lse, unique_lse)
        cascade_out = (
            shared_out * torch.exp(shared_lse - normalizer).unsqueeze(-1)
            + unique_out * torch.exp(unique_lse - normalizer).unsqueeze(-1)
        )

        full_out, _ = attention(
            torch.cat((expanded_shared_k, unique_k), dim=1),
            torch.cat((expanded_shared_v, unique_v), dim=1),
        )
        torch.testing.assert_close(cascade_out, full_out, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
