"""Synthetic gfx942 controls for DSA kpool tail handling.

The kernel-level mixed-dtype probe intentionally calls AITER directly.  The
production DSA backend still rejects ``index_kpool > 1`` with AITER; the final
test pins that admission error so the direct probe cannot be mistaken for an
enabled end-to-end route.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.dsa.dsa_backend_kpool import (
    DeepseekSparseAttnBackendKPoolMixin,
)
from sglang.srt.layers.attention.dsa.kpool_fp8_index import (
    topk_from_pooled_history_logits,
)
from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd")


_RUNNABLE = is_hip() and torch.cuda.is_available()
if _RUNNABLE:
    try:
        from aiter.mla import mla_decode_fwd
        from aiter.ops.attention import (
            get_mla_metadata_info_v1,
            get_mla_metadata_v1,
        )
        from sglang.kernels.ops.quantization.fp8_kernel import is_fp8_fnuz
    except Exception:
        _RUNNABLE = False


def _run_mla_decode(q, kv_cache, page_table, topk, fp8_dtype):
    batch_size, num_heads, head_dim = q.shape
    value_dim = 512
    output = torch.empty(
        (batch_size, num_heads, value_dim),
        dtype=torch.bfloat16,
        device=q.device,
    )
    qo_indptr = torch.arange(
        batch_size + 1, dtype=torch.int32, device=q.device
    )
    valid = page_table != -1
    kv_indptr = torch.zeros(
        batch_size + 1, dtype=torch.int32, device=q.device
    )
    kv_indptr[1:] = valid.sum(dim=1).cumsum(dim=0)
    kv_indices = page_table[valid].to(torch.int32).reshape(-1)
    kv_last_page_lens = torch.ones(
        batch_size, dtype=torch.int32, device=q.device
    )

    sizes = get_mla_metadata_info_v1(
        batch_size,
        1,
        num_heads,
        q.dtype,
        kv_cache.dtype,
        is_sparse=True,
        fast_mode=False,
        num_kv_splits=64,
        intra_batch_mode=True,
    )
    buffers = [
        torch.empty(size, dtype=dtype, device=q.device)
        for size, dtype in sizes
    ]
    (
        work_metadata,
        work_indptr,
        work_info_set,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
    ) = buffers
    get_mla_metadata_v1(
        qo_indptr,
        kv_indptr,
        kv_last_page_lens,
        num_heads,
        1,
        False,
        work_metadata,
        work_info_set,
        work_indptr,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
        page_size=1,
        kv_granularity=16,
        max_seqlen_qo=1,
        uni_seqlen_qo=1,
        fast_mode=False,
        topk=topk,
        max_split_per_batch=64,
        intra_batch_mode=True,
        dtype_q=q.dtype,
        dtype_kv=kv_cache.dtype,
    )
    mla_decode_fwd(
        q,
        kv_cache,
        output,
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_lens,
        1,
        sm_scale=head_dim**-0.5,
        q_scale=None,
        kv_scale=(
            torch.ones((), dtype=torch.float32, device=q.device)
            if kv_cache.dtype == fp8_dtype
            else None
        ),
        work_meta_data=work_metadata,
        work_indptr=work_indptr,
        work_info_set=work_info_set,
        reduce_indptr=reduce_indptr,
        reduce_final_map=reduce_final_map,
        reduce_partial_map=reduce_partial_map,
        intra_batch_mode=True,
        num_kv_splits=64,
    )
    return output


def _reference_decode(q, kv_cache, page_table, value_dim=512):
    outputs = []
    for row, query in enumerate(q):
        indices = page_table[row][page_table[row] != -1].long()
        keys = kv_cache[indices, 0, 0].to(torch.bfloat16)
        scores = torch.einsum("hd,kd->hk", query, keys)
        scores *= query.shape[-1] ** -0.5
        weights = scores.softmax(dim=-1)
        outputs.append(torch.einsum("hk,kd->hd", weights, keys[:, :value_dim]))
    return torch.stack(outputs)


@unittest.skipUnless(_RUNNABLE, "requires HIP with AITER MLA and sgl_kernel")
class TestDsaKpoolRocmControl(CustomTestCase):
    def setUp(self):
        torch.manual_seed(20260910)
        self.device = torch.device("cuda")
        self.fp8_dtype = (
            torch.float8_e4m3fnuz if is_fp8_fnuz() else torch.float8_e4m3fn
        )

    def _make_q_and_cache(self, cache_tokens):
        q = torch.randn(
            2, 16, 576, dtype=torch.bfloat16, device=self.device
        )
        cache = torch.randn(
            2 * cache_tokens,
            1,
            1,
            576,
            dtype=torch.bfloat16,
            device=self.device,
        ).to(self.fp8_dtype)
        return q, cache

    def test_kpool_one_bf16_query_fp8_kv_matches_reference(self):
        q, cache = self._make_q_and_cache(128)
        page_table = torch.stack(
            [torch.randperm(128, device=self.device) for _ in range(2)]
        ).to(torch.int32)

        actual = _run_mla_decode(q, cache, page_table, 128, self.fp8_dtype)
        expected = _reference_decode(q, cache, page_table)

        torch.testing.assert_close(
            actual, expected, atol=0.05, rtol=0.05, check_dtype=False
        )

    def test_kpool_four_tail_indices_and_mixed_dtype_output(self):
        pool_size = 4
        group_topk = 128
        topk = group_topk * pool_size
        group_lengths = torch.tensor([3, 130], device=self.device)
        seq_lens = torch.tensor([13, 521], device=self.device)
        cache_tokens = int(seq_lens.max().item())
        page_table = (
            torch.arange(2 * cache_tokens, device=self.device)
            .reshape(2, cache_tokens)
        ).to(torch.int32)
        scores = torch.randn(2, 130, device=self.device)

        actual = topk_from_pooled_history_logits(
            scores,
            group_lengths,
            pool_size,
            topk,
            page_table=page_table,
            seq_lens=seq_lens,
        )
        self.assertEqual(actual.shape, (2, topk + pool_size - 1))

        for row in range(2):
            length = int(group_lengths[row].item())
            history_len = min(length * pool_size, topk)
            expected_groups = (
                torch.arange(length, device=self.device)
                if length <= group_topk
                else torch.topk(scores[row, :length], group_topk).indices
            )
            expected_history = page_table[
                row,
                torch.repeat_interleave(expected_groups * pool_size, pool_size)
                + torch.arange(pool_size, device=self.device).repeat(
                    expected_groups.numel()
                ),
            ]
            self.assertEqual(
                sorted(actual[row, :history_len].tolist()),
                sorted(expected_history.tolist()),
            )

            valid = actual[row][actual[row] != -1]
            expected_tail = page_table[
                row,
                length * pool_size : int(seq_lens[row]),
            ]
            self.assertEqual(
                valid[-len(expected_tail) :].tolist(),
                expected_tail.tolist(),
            )

        q, cache = self._make_q_and_cache(cache_tokens)
        output = _run_mla_decode(q, cache, actual, topk, self.fp8_dtype)
        expected = _reference_decode(q, cache, actual)
        torch.testing.assert_close(
            output, expected, atol=0.05, rtol=0.05, check_dtype=False
        )

    def test_kpool_four_aiter_admission_is_precise(self):
        backend = SimpleNamespace(
            dsa_index_kpool=4,
            device_sm_major=torch.cuda.get_device_capability()[0],
        )
        mixin = DeepseekSparseAttnBackendKPoolMixin()
        mixin.__dict__.update(backend.__dict__)
        topk_indices = torch.zeros((1, 515), dtype=torch.int32, device="cuda")

        with self.assertRaisesRegex(
            NotImplementedError,
            "index_kpool > 1 appends tail tokens.*only supported by the "
            "FA3/TileLang/TRTLLM DSA decode backend",
        ):
            mixin._check_kpool_tail_backend(
                topk_indices, "aiter", "decode"
            )


if __name__ == "__main__":
    unittest.main()
