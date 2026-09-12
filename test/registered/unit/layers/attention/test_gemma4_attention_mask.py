"""Regression tests for compact Gemma 4 image-attention metadata."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers.attention.triton_backend import TritonAttnBackend  # noqa: E402
from sglang.srt.model_executor.forward_batch_info import ForwardMode  # noqa: E402
from sglang.srt.models.gemma4_mm import Gemma4ForConditionalGeneration  # noqa: E402

register_cpu_ci(est_time=30, suite="base-a-test-cpu")


class _FakeTritonBackend:
    def __init__(self):
        self.forward_metadata = SimpleNamespace(
            custom_mask=None,
            mask_indptr=None,
            image_span_indptr=None,
            image_span_begin=None,
            image_span_end=None,
        )


class _FakeImage:
    def __init__(self, offsets):
        self.offsets = offsets

    @staticmethod
    def is_image():
        return True


class TestGemma4AttentionMask(CustomTestCase):
    def _prepare(self, backend, forward_batch, num_tokens):
        with (
            patch("sglang.srt.models.gemma4_mm.TritonAttnBackend", _FakeTritonBackend),
            patch("sglang.srt.models.gemma4_mm.get_attn_backend", return_value=backend),
        ):
            Gemma4ForConditionalGeneration.prepare_attn_masks(
                None,
                forward_batch,
                input_ids=torch.arange(num_tokens),
                mask_dtype=torch.bool,
            )

    def test_image_mask_uses_compact_spans_on_sliding_attention(self):
        backend = _FakeTritonBackend()
        forward_batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            batch_size=1,
            extend_seq_lens_cpu=[8],
            extend_prefix_lens_cpu=[0],
            extend_seq_lens=torch.tensor([8]),
            extend_prefix_lens=torch.tensor([0]),
            mm_inputs=[SimpleNamespace(mm_items=[_FakeImage(((1, 3), (5, 6)))])],
        )

        self._prepare(backend, forward_batch, 8)

        metadata = backend.forward_metadata
        torch.testing.assert_close(
            metadata.image_span_indptr, torch.tensor([0, 2], dtype=torch.int64)
        )
        torch.testing.assert_close(
            metadata.image_span_begin, torch.tensor([1, 5], dtype=torch.int64)
        )
        torch.testing.assert_close(
            metadata.image_span_end, torch.tensor([4, 7], dtype=torch.int64)
        )
        self.assertIsNone(metadata.custom_mask)

        full_spans = TritonAttnBackend._get_image_spans_for_layer(
            backend, SimpleNamespace(sliding_window_size=-1)
        )
        self.assertTrue(all(value is None for value in full_spans.values()))
        sliding_spans = TritonAttnBackend._get_image_spans_for_layer(
            backend, SimpleNamespace(sliding_window_size=1024)
        )
        self.assertEqual(sum(t.numel() for t in sliding_spans.values()), 6)
        self.assertLess(6, 8 * 8)

    def test_mixed_batch_indptr_and_split_image_fallback(self):
        backend = _FakeTritonBackend()
        forward_batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            batch_size=3,
            extend_seq_lens_cpu=[8, 4, 5],
            extend_prefix_lens_cpu=[0, 4, 10],
            extend_seq_lens=torch.tensor([8, 4, 5]),
            extend_prefix_lens=torch.tensor([0, 4, 10]),
            mm_inputs=[
                None,
                SimpleNamespace(mm_items=[_FakeImage(((4, 7),))]),
                # Intersects this chunk but is not fully contained, so it must
                # remain causal rather than exposing a partial image span.
                SimpleNamespace(mm_items=[_FakeImage(((8, 12),))]),
            ],
        )

        self._prepare(backend, forward_batch, 17)

        metadata = backend.forward_metadata
        torch.testing.assert_close(
            metadata.image_span_indptr, torch.tensor([0, 0, 1, 1])
        )
        torch.testing.assert_close(metadata.image_span_begin, torch.tensor([4]))
        torch.testing.assert_close(metadata.image_span_end, torch.tensor([8]))


if __name__ == "__main__":
    unittest.main()
