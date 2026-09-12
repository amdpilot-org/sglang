"""Regression tests for bounded DSA kpool plan-path MQA logits."""

from types import SimpleNamespace
from unittest import mock

import pytest

torch = pytest.importorskip("torch")

from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.layers.attention.dsa.dsa_indexer_kpool import (  # noqa: E402
    IndexerKPool,
)
from sglang.srt.layers.attention.dsa.dsa_topk_backend import (  # noqa: E402
    TopkTransformMethod,
)
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

GIB = 2**30


def _indexer():
    indexer = IndexerKPool.__new__(IndexerKPool)
    torch.nn.Module.__init__(indexer)
    indexer.index_topk = 8
    indexer.index_kpool = 4
    return indexer


@pytest.mark.parametrize(
    "num_q,num_k,free_mem,total_mem,expected",
    [
        (1_000, 1_000, None, None, (False, 0)),
        (8_192, 348_032, 14 * GIB, 95 * GIB, (True, 14 * GIB // 4)),
        (8_192, 348_032, 80 * GIB, 95 * GIB, (False, 20 * GIB)),
        (8_192, 1_310_720, 80 * GIB, 95 * GIB, (True, 20 * GIB)),
        (8_192, 348_032, 0, 95 * GIB, (True, 1)),
    ],
)
def test_mqa_logits_budget_boundaries(num_q, num_k, free_mem, total_mem, expected):
    with mock.patch("torch.cuda.mem_get_info") as mem_get_info:
        if free_mem is not None:
            mem_get_info.return_value = (free_mem, total_mem)
        result = _indexer()._should_chunk_mqa_logits(
            num_q, num_k, torch.device("cuda")
        )
    assert result == expected
    assert mem_get_info.call_count == (free_mem is not None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires one GPU")
def test_kpool_plan_chunks_rows_and_preserves_results():
    device = torch.device("cuda")
    n_real, total_q, total_k, head_dim = 5, 7, 12, 8
    starts = torch.tensor([0, 0, 6, 6, 6], dtype=torch.int32, device=device)
    ends = torch.tensor([5, 6, 9, 11, 12], dtype=torch.int32, device=device)
    pool_lens = ends - starts
    plan = SimpleNamespace(
        seq_lens_expanded=pool_lens * 4,
        pooled_seq_lens_expanded=pool_lens,
        ragged_q_ks=starts,
        ragged_q_ke=ends,
        ragged_total_k_rows=total_k,
        ragged_k_u8=torch.zeros((total_k, head_dim), dtype=torch.uint8, device=device),
        ragged_k_scale=torch.ones(total_k, dtype=torch.float32, device=device),
        ragged_concat_page_table=torch.zeros(1, dtype=torch.int32, device=device),
        ragged_paged_page_table=None,
        ragged_paged_page_table_row_index=None,
        ragged_groups=(
            SimpleNamespace(q_start=0, q_len=2, k_start=0, k_rows=6),
            SimpleNamespace(q_start=2, q_len=3, k_start=6, k_rows=6),
        ),
    )
    metadata = SimpleNamespace(
        attn_metadata=SimpleNamespace(
            kpool_extend_plan=plan, topk_indices_offset=None
        ),
        topk_transform_method=TopkTransformMethod.RAGGED,
    )
    q = torch.zeros(
        (total_q, 1, head_dim), dtype=torch.float8_e4m3fn, device=device
    )
    weights = torch.ones((total_q, 1, 1), dtype=torch.float32, device=device)

    def fake_logits(q_chunk, kv, weights_chunk, ks, ke, clean_logits):
        cols = torch.arange(total_k, dtype=torch.float32, device=device)
        return ke.float().unsqueeze(1) * 100 + cols.unsqueeze(0)

    def fake_topk(logits, pool_lens_chunk, **kwargs):
        rows = torch.arange(logits.shape[0], device=device)
        result = torch.full(
            (logits.shape[0], 11), -1, dtype=torch.int32, device=device
        )
        result[:, :2] = torch.stack(
            (rows, pool_lens_chunk.to(torch.int64)), dim=1
        ).to(torch.int32)
        return result

    def run(chunk_rows):
        indexer = _indexer()
        budget = chunk_rows * total_k * 4
        with (
            mock.patch.object(
                IndexerKPool,
                "_should_chunk_mqa_logits",
                return_value=(chunk_rows < n_real, budget),
            ),
            mock.patch.object(IndexerKPool, "_get_index_k_read_buffer"),
            mock.patch.object(
                IndexerKPool, "_topk_from_kpool_logits", side_effect=fake_topk
            ),
            mock.patch(
                "sglang.srt.layers.attention.dsa.dsa_indexer_kpool.get_token_to_kv_pool"
            ),
            mock.patch(
                "sglang.srt.layers.attention.dsa.kpool_fp8_index."
                "gather_index_k_scale_prefix_into"
            ),
            mock.patch(
                "sglang.srt.layers.attention.dsa.dsa_indexer_kpool.deep_gemm",
                SimpleNamespace(fp8_mqa_logits=mock.Mock(side_effect=fake_logits)),
                create=True,
            ) as deep_gemm,
            envs.SGLANG_DSA_FUSE_TOPK.override(False),
        ):
            result = indexer._get_topk_ragged_kpool_plan(
                mock.Mock(), 0, q, weights, metadata
            )
        return result, deep_gemm.fp8_mqa_logits.call_count

    dense, dense_calls = run(n_real)
    chunked, chunked_calls = run(2)
    assert dense_calls == 1
    assert chunked_calls == 2
    assert torch.equal(dense[:n_real, 1], chunked[:n_real, 1])
    assert torch.all(chunked[n_real:] == -1)
