"""Coverage for the AITER FP8-Q unified-attention decode path."""

import json
import math
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=120, suite="stage-b-test-1-gpu-small-amd-mi35x")

_RUNNABLE = is_hip()
if _RUNNABLE:
    from aiter.ops.triton.attention.unified_attention import unified_attention

    import sglang.srt.layers.attention.aiter_backend as aiter_backend
    from sglang.kernels.ops.quantization.fp8_kernel import (
        fp8_dtype,
        scaled_fp8_quant,
    )
    from sglang.srt.layers.attention.aiter_backend import AiterAttnBackend


class _FakeKVPool:
    def __init__(self, k_cache, v_cache):
        self.k_cache = k_cache
        self.v_cache = v_cache

    def get_kv_buffer(self, _layer_id):
        return self.k_cache, self.v_cache

    def get_key_buffer(self, _layer_id):
        return self.k_cache


@unittest.skipUnless(_RUNNABLE, "requires HIP with AITER unified attention")
class TestAiterFP8QUnifiedAttention(CustomTestCase):
    def _run_unified_attention(
        self,
        q,
        k,
        v,
        output,
        seq_len,
        softmax_scale,
        q_descale=None,
        k_descale=None,
        v_descale=None,
    ):
        batch = q.shape[0]
        page_size = k.shape[1]
        pages_per_seq = seq_len // page_size
        block_table = torch.arange(
            batch * pages_per_seq, dtype=torch.int32, device=q.device
        ).view(batch, pages_per_seq)

        unified_attention(
            q=q,
            k=k,
            v=v,
            out=output,
            cu_seqlens_q=torch.arange(
                batch + 1, dtype=torch.int32, device=q.device
            ),
            seqused_k=torch.full(
                (batch,), seq_len, dtype=torch.int32, device=q.device
            ),
            max_seqlen_q=1,
            max_seqlen_k=seq_len,
            softmax_scale=softmax_scale,
            causal=True,
            window_size=(-1, -1),
            block_table=block_table,
            softcap=0,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            sinks=None,
        )

    def _reference_attention(self, q, k, v, softmax_scale):
        scores = torch.einsum("bhd,btd->bht", q.float(), k[:, :, 0, :].float())
        scores = scores * softmax_scale
        probabilities = torch.softmax(scores, dim=-1)
        return torch.einsum(
            "bht,btd->bhd", probabilities, v[:, :, 0, :].float()
        )

    def _comparison_metrics(self, actual, expected):
        actual = actual.float()
        expected = expected.float()
        difference = actual - expected
        mismatch = difference.abs() > 0.15 + 0.15 * expected.abs()
        return {
            "max_abs": difference.abs().max().item(),
            "mean_abs": difference.abs().mean().item(),
            "mismatch_fraction": mismatch.float().mean().item(),
            "cosine": torch.nn.functional.cosine_similarity(
                actual.flatten(), expected.flatten(), dim=0
            ).item(),
        }

    def _timed_attention(self, run_attention, warmups=3, measurements=5):
        for _ in range(warmups):
            run_attention()
        torch.cuda.synchronize()

        samples = []
        for _ in range(measurements):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            run_attention()
            end.record()
            torch.cuda.synchronize()
            samples.append(start.elapsed_time(end))
        return samples

    def _make_backend_case(self, branch, kv_cache_dtype=None):
        if kv_cache_dtype is None:
            kv_cache_dtype = fp8_dtype
        batch, num_q_heads, num_kv_heads, head_dim = 2, 2, 1, 8
        device = "cuda"
        scale = torch.tensor([0.02], dtype=torch.float32, device=device)
        k_cache = torch.zeros(
            batch,
            1,
            num_kv_heads,
            head_dim,
            dtype=kv_cache_dtype,
            device=device,
        )
        v_cache = torch.zeros_like(k_cache)

        backend = object.__new__(AiterAttnBackend)
        backend.use_mla = branch == "mla"
        backend.kv_cache_is_vectorized_5d = branch == "vectorized"
        backend.use_triton_unified_attention = branch != "legacy"
        backend.kv_cache_dtype = kv_cache_dtype
        backend.input_dtype = torch.bfloat16
        backend.page_size = 1
        backend.scale = head_dim**-0.5
        backend.logits_soft_cap = 0.0
        backend.k_scale = scale
        backend.v_scale = scale
        backend.workspace_buffer = torch.empty(1, device=device)
        backend.max_num_partitions = 1
        backend.kv_last_page_len = torch.ones(batch, dtype=torch.int32, device=device)
        backend.token_to_kv_pool = _FakeKVPool(k_cache, v_cache)
        backend.forward_metadata = SimpleNamespace(
            kv_indices=torch.arange(batch, dtype=torch.int32, device=device).view(
                batch, 1
            ),
            swa_page_table=None,
            qo_indptr=torch.arange(batch + 1, dtype=torch.int32, device=device),
            kv_indptr=torch.arange(batch + 1, dtype=torch.int32, device=device),
            kv_last_page_len=backend.kv_last_page_len,
            max_q_len=1,
            work_metadata=None,
            work_indptr=None,
            work_info_set=None,
            reduce_indptr=None,
            reduce_final_map=None,
            reduce_partial_map=None,
            num_kv_splits=None,
        )
        backend._mla_decode_fwd_with_head_pad = mock.Mock(
            return_value=torch.empty(
                batch,
                num_q_heads,
                head_dim,
                dtype=torch.bfloat16,
                device=device,
            )
        )
        backend._get_aiter_paged_ragged_kv_cache_dtype = mock.Mock(
            return_value="fp8_e4m3"
        )

        layer = SimpleNamespace(
            layer_id=0,
            tp_q_head_num=num_q_heads,
            tp_k_head_num=num_kv_heads,
            tp_v_head_num=num_kv_heads,
            qk_head_dim=head_dim,
            v_head_dim=head_dim,
            k_scale=scale,
            v_scale=scale,
            sliding_window_size=-1,
            scaling=head_dim**-0.5,
            logit_cap=0.0,
        )
        forward_batch = SimpleNamespace(
            batch_size=batch,
            seq_lens=torch.ones(batch, dtype=torch.int32, device=device),
        )
        q = torch.randn(
            batch,
            num_q_heads,
            head_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        return backend, layer, forward_batch, q

    def test_q_quantization_is_isolated_to_unified_attention(self):
        for branch in ("mla", "vectorized", "unified", "legacy"):
            with self.subTest(branch=branch):
                backend, layer, forward_batch, q = self._make_backend_case(branch)
                original_q = q.reshape(q.shape[0], -1).clone()
                sentinel_q = torch.full(
                    original_q.shape, 7, dtype=fp8_dtype, device=original_q.device
                )

                with (
                    mock.patch.object(
                        aiter_backend,
                        "scaled_fp8_quant",
                        return_value=(sentinel_q, layer.k_scale),
                    ) as quant,
                    mock.patch.object(aiter_backend, "unified_attention") as unified,
                    mock.patch.object(
                        aiter_backend, "forward_decode_vectorized_5d"
                    ) as vectorized,
                    mock.patch.object(
                        aiter_backend, "paged_attention_ragged"
                    ) as legacy,
                ):
                    output = backend.forward_decode(
                        q, None, None, layer, forward_batch, save_kv_cache=False
                    )

                self.assertEqual(output.numel(), q.numel())
                if branch == "unified":
                    quant.assert_called_once()
                    self.assertIs(quant.call_args.args[1], layer.k_scale)
                    self.assertIs(unified.call_args.kwargs["q_descale"], layer.k_scale)
                    torch.testing.assert_close(
                        unified.call_args.kwargs["q"].reshape(q.shape[0], -1),
                        sentinel_q,
                    )
                else:
                    quant.assert_not_called()
                    if branch == "mla":
                        observed_q = (
                            backend._mla_decode_fwd_with_head_pad.call_args.args[0]
                        )
                    elif branch == "vectorized":
                        observed_q = vectorized.call_args.args[1]
                    else:
                        observed_q = legacy.call_args.args[2]
                    torch.testing.assert_close(
                        observed_q.reshape(q.shape[0], -1), original_q
                    )

    def test_bf16_kv_keeps_bf16_q(self):
        backend, layer, forward_batch, q = self._make_backend_case(
            "unified", torch.bfloat16
        )
        original_q = q.reshape(q.shape[0], -1).clone()

        with (
            mock.patch.object(aiter_backend, "scaled_fp8_quant") as quant,
            mock.patch.object(aiter_backend, "unified_attention") as unified,
        ):
            output = backend.forward_decode(
                q, None, None, layer, forward_batch, save_kv_cache=False
            )

        self.assertEqual(output.numel(), q.numel())
        quant.assert_not_called()
        self.assertIsNone(unified.call_args.kwargs["q_descale"])
        observed_q = unified.call_args.kwargs["q"]
        self.assertEqual(observed_q.dtype, torch.bfloat16)
        torch.testing.assert_close(
            observed_q.reshape(q.shape[0], -1),
            original_q,
        )

    def test_fp8_q_kv_matches_bf16_reference_at_decode_shape(self):
        # This is the per-TP-rank production shape used by the full trace.
        batch, num_q_heads, num_kv_heads = 4, 16, 1
        seq_len, head_dim, page_size = 8192, 256, 16
        device = "cuda"
        torch.manual_seed(0)

        base = torch.randn(batch, head_dim, device=device, dtype=torch.float32)
        q = (
            base[:, None, :].expand(-1, num_q_heads, -1)
            + 0.01
            * torch.randn(
                batch, num_q_heads, head_dim, device=device, dtype=torch.float32
            )
        ).to(torch.bfloat16)
        k = 0.1 * torch.randn(
            batch,
            seq_len,
            num_kv_heads,
            head_dim,
            device=device,
            dtype=torch.float32,
        )
        k[:, 0, 0, :] = 2 * base
        k = k.to(torch.bfloat16)
        v = (
            0.75
            + 0.25
            * torch.randn(
                batch,
                seq_len,
                num_kv_heads,
                head_dim,
                device=device,
                dtype=torch.float32,
            )
        ).to(torch.bfloat16)

        fp8_max = torch.finfo(fp8_dtype).max
        k_scale = (k.abs().float().amax() / fp8_max).clamp(min=1e-9).view(1)
        v_scale = (v.abs().float().amax() / fp8_max).clamp(min=1e-9).view(1)
        q_fp8, _ = scaled_fp8_quant(q.reshape(batch, -1), k_scale)
        k_fp8, _ = scaled_fp8_quant(k.reshape(-1, head_dim), k_scale)
        v_fp8, _ = scaled_fp8_quant(v.reshape(-1, head_dim), v_scale)

        q_fp8 = q_fp8.view(batch, num_q_heads, head_dim)
        k_fp8 = k_fp8.view(-1, page_size, num_kv_heads, head_dim)
        v_fp8 = v_fp8.view(-1, page_size, num_kv_heads, head_dim)
        pages_per_seq = seq_len // page_size
        block_table = torch.arange(
            batch * pages_per_seq, dtype=torch.int32, device=device
        ).view(batch, pages_per_seq)
        output = torch.empty_like(q, dtype=torch.bfloat16)

        unified_attention(
            q=q_fp8,
            k=k_fp8,
            v=v_fp8,
            out=output,
            cu_seqlens_q=torch.arange(batch + 1, dtype=torch.int32, device=device),
            seqused_k=torch.full((batch,), seq_len, dtype=torch.int32, device=device),
            max_seqlen_q=1,
            max_seqlen_k=seq_len,
            softmax_scale=1 / math.sqrt(head_dim),
            causal=True,
            window_size=(-1, -1),
            block_table=block_table,
            softcap=0,
            q_descale=k_scale,
            k_descale=k_scale,
            v_descale=v_scale,
            sinks=None,
        )

        scores = torch.einsum(
            "bhd,btd->bht", q.float(), k[:, :, 0, :].float()
        ) / math.sqrt(head_dim)
        expected = torch.einsum(
            "bht,btd->bhd",
            torch.softmax(scores, dim=-1),
            v[:, :, 0, :].float(),
        )
        actual = output.float()

        self.assertTrue(bool(torch.isfinite(actual).all()))
        self.assertGreater(expected.abs().mean().item(), 0.25)
        mismatch = (actual - expected).abs() > 0.15 + 0.15 * expected.abs()
        mismatch_fraction = mismatch.float().mean().item()
        self.assertLess(
            mismatch_fraction,
            0.005,
            f"FP8 mismatch fraction {mismatch_fraction:.4%} exceeds 0.5%; "
            f"max abs diff={(actual - expected).abs().max().item():.6f}",
        )
        cosine = torch.nn.functional.cosine_similarity(
            actual.flatten(), expected.flatten(), dim=0
        ).item()
        self.assertGreater(cosine, 0.99)

    def test_fp8_context_growth_matches_bf16_and_dequantized_references(self):
        batch, num_q_heads, num_kv_heads = 4, 16, 1
        head_dim, page_size = 256, 16
        sequence_lengths = (256, 1024, 4096, 8192, 16384)
        max_sequence_length = sequence_lengths[-1]
        device = "cuda"
        softmax_scale = 1 / math.sqrt(head_dim)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

        q_bf16 = (
            0.1
            * torch.randn(
                batch,
                num_q_heads,
                head_dim,
                device=device,
                dtype=torch.float32,
            )
        ).to(torch.bfloat16)
        k_bf16 = (
            0.1
            * torch.randn(
                batch,
                max_sequence_length,
                num_kv_heads,
                head_dim,
                device=device,
                dtype=torch.float32,
            )
        ).to(torch.bfloat16)
        v_bf16 = (
            0.75
            + 0.25
            * torch.randn(
                batch,
                max_sequence_length,
                num_kv_heads,
                head_dim,
                device=device,
                dtype=torch.float32,
            )
        ).to(torch.bfloat16)

        fp8_max = torch.finfo(fp8_dtype).max
        q_scale = (q_bf16.abs().float().amax() / fp8_max).clamp(min=1e-9).view(1)
        k_scale = (k_bf16.abs().float().amax() / fp8_max).clamp(min=1e-9).view(1)
        v_scale = (v_bf16.abs().float().amax() / fp8_max).clamp(min=1e-9).view(1)
        q_fp8, _ = scaled_fp8_quant(q_bf16.reshape(batch, -1), q_scale)
        q_fp8 = q_fp8.view(batch, num_q_heads, head_dim)

        output_bf16 = torch.empty_like(q_bf16)
        output_fp8 = torch.empty_like(q_bf16)
        records = []
        bf16_outputs = []

        for sequence_length in sequence_lengths:
            with self.subTest(sequence_length=sequence_length):
                k_prefix = k_bf16[:, :sequence_length]
                v_prefix = v_bf16[:, :sequence_length]
                k_prefix_fp8, _ = scaled_fp8_quant(
                    k_prefix.reshape(-1, head_dim), k_scale
                )
                v_prefix_fp8, _ = scaled_fp8_quant(
                    v_prefix.reshape(-1, head_dim), v_scale
                )
                k_prefix_fp8 = k_prefix_fp8.view(
                    -1, page_size, num_kv_heads, head_dim
                )
                v_prefix_fp8 = v_prefix_fp8.view(
                    -1, page_size, num_kv_heads, head_dim
                )

                def run_bf16():
                    self._run_unified_attention(
                        q_bf16,
                        k_prefix.reshape(
                            -1, page_size, num_kv_heads, head_dim
                        ),
                        v_prefix.reshape(
                            -1, page_size, num_kv_heads, head_dim
                        ),
                        output_bf16,
                        sequence_length,
                        softmax_scale,
                    )

                def run_fp8():
                    self._run_unified_attention(
                        q_fp8,
                        k_prefix_fp8,
                        v_prefix_fp8,
                        output_fp8,
                        sequence_length,
                        softmax_scale,
                        q_descale=q_scale,
                        k_descale=k_scale,
                        v_descale=v_scale,
                    )

                run_bf16()
                run_fp8()
                torch.cuda.synchronize()
                bf16_outputs.append(output_bf16.clone())

                bf16_reference = self._reference_attention(
                    q_bf16, k_prefix, v_prefix, softmax_scale
                )
                q_dequantized = q_fp8.float() * q_scale
                k_dequantized = k_prefix_fp8.view(
                    batch, sequence_length, num_kv_heads, head_dim
                ).float() * k_scale
                v_dequantized = v_prefix_fp8.view(
                    batch, sequence_length, num_kv_heads, head_dim
                ).float() * v_scale
                dequantized_reference = self._reference_attention(
                    q_dequantized,
                    k_dequantized,
                    v_dequantized,
                    softmax_scale,
                )

                comparisons = {
                    "bf16_vs_reference": self._comparison_metrics(
                        output_bf16, bf16_reference
                    ),
                    "fp8_vs_dequantized_reference": self._comparison_metrics(
                        output_fp8, dequantized_reference
                    ),
                    "fp8_vs_bf16": self._comparison_metrics(
                        output_fp8, output_bf16
                    ),
                }
                bf16_samples = self._timed_attention(run_bf16)
                fp8_samples = self._timed_attention(run_fp8)
                record = {
                    "sequence_length": sequence_length,
                    "comparisons": comparisons,
                    "timing_ms": {
                        "bf16_median": sorted(bf16_samples)[
                            len(bf16_samples) // 2
                        ],
                        "fp8_median": sorted(fp8_samples)[
                            len(fp8_samples) // 2
                        ],
                        "bf16_samples": bf16_samples,
                        "fp8_samples": fp8_samples,
                    },
                }
                records.append(record)

                self.assertTrue(
                    bool(torch.isfinite(output_bf16).all()),
                    "BF16 output contains non-finite values",
                )
                self.assertTrue(
                    bool(torch.isfinite(output_fp8).all()),
                    "FP8 output contains non-finite values",
                )
                self.assertGreater(bf16_reference.abs().mean().item(), 0.25)
                for comparison_name, comparison in comparisons.items():
                    self.assertLess(
                        comparison["mismatch_fraction"],
                        0.005,
                        f"{comparison_name} mismatch fraction exceeds 0.5%",
                    )
                    self.assertGreater(
                        comparison["cosine"],
                        0.99,
                        f"{comparison_name} cosine similarity is too low",
                    )

                print(json.dumps(record, sort_keys=True), flush=True)

        output_change = (
            bf16_outputs[0].float() - bf16_outputs[-1].float()
        ).abs().max().item()
        self.assertGreater(
            output_change,
            1e-3,
            "Synthetic prefixes did not exercise context-length growth",
        )


if __name__ == "__main__":
    unittest.main()
