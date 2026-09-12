"""Counterexamples for request-grouped DSA kpool logits chunking."""

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


def _indexer():
    indexer = IndexerKPool.__new__(IndexerKPool)
    torch.nn.Module.__init__(indexer)
    indexer.index_topk = 8
    indexer.index_kpool = 4
    return indexer


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires one GPU")
@pytest.mark.parametrize(
    "method", [TopkTransformMethod.RAGGED, TopkTransformMethod.PAGED]
)
def test_grouped_chunking_preserves_global_mapping_and_budget(method):
    device = torch.device("cuda")
    n_real, total_q, total_k, dim = 5, 7, 12, 8
    starts = torch.tensor([0, 0, 6, 6, 6], dtype=torch.int32, device=device)
    ends = torch.tensor([5, 6, 9, 11, 12], dtype=torch.int32, device=device)
    pool_lens = ends - starts
    offsets = torch.tensor([101, 102, 201, 202, 203], dtype=torch.int32, device=device)
    row_index = torch.tensor([3, 3, 9, 9, 9], dtype=torch.int32, device=device)
    page_table = torch.arange(80, dtype=torch.int32, device=device).reshape(10, 8)
    plan = SimpleNamespace(
        seq_lens_expanded=pool_lens * 4,
        pooled_seq_lens_expanded=pool_lens,
        ragged_q_ks=starts,
        ragged_q_ke=ends,
        ragged_total_k_rows=total_k,
        ragged_k_u8=torch.zeros((total_k, dim), dtype=torch.uint8, device=device),
        ragged_k_scale=torch.ones(total_k, dtype=torch.float32, device=device),
        ragged_concat_page_table=torch.zeros(1, dtype=torch.int32, device=device),
        ragged_paged_page_table=page_table,
        ragged_paged_page_table_row_index=row_index,
        ragged_groups=(
            SimpleNamespace(q_start=0, q_len=2, k_start=0, k_rows=6),
            SimpleNamespace(q_start=2, q_len=3, k_start=6, k_rows=6),
        ),
    )
    metadata = SimpleNamespace(
        attn_metadata=SimpleNamespace(
            kpool_extend_plan=plan, topk_indices_offset=offsets
        ),
        topk_transform_method=method,
    )
    q = torch.zeros((total_q, 1, dim), dtype=torch.float8_e4m3fn, device=device)
    weights = torch.ones((total_q, 1, 1), dtype=torch.float32, device=device)
    # RAGGED exercises a budget smaller than one concatenated-K row. PAGED uses
    # one concatenated row so it reaches the later chunk's global table mapping.
    budget = (6 if method == TopkTransformMethod.RAGGED else total_k) * 4
    calls = []

    def fake_logits(q_chunk, kv, weights_chunk, ks, ke, clean_logits):
        calls.append((q_chunk.shape[0], kv[0].shape[0]))
        return torch.zeros(
            (q_chunk.shape[0], kv[0].shape[0]), dtype=torch.float32, device=device
        )

    def fake_topk(logits, lens, **kw):
        cols = [lens, kw["seq_lens"], kw["row_starts"]]
        if method == TopkTransformMethod.RAGGED:
            cols.append(kw["topk_offsets"])
            assert kw["page_table"] is None
            assert kw["page_table_row_index"] is None
        else:
            cols.append(kw["page_table_row_index"])
            assert kw["page_table"].data_ptr() == page_table.data_ptr()
            assert kw["topk_offsets"] is None
        assert logits.numel() * logits.element_size() <= budget
        result = torch.full(
            (logits.shape[0], 11), -1, dtype=torch.int32, device=device
        )
        result[:, :4] = torch.stack(cols, dim=1).to(torch.int32)
        return result

    with (
        mock.patch.object(
            IndexerKPool, "_should_chunk_mqa_logits", return_value=(True, budget)
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
        ),
        envs.SGLANG_DSA_FUSE_TOPK.override(True),
    ):
        result = _indexer()._get_topk_ragged_kpool_plan(
            mock.Mock(), 0, q, weights, metadata
        )

    expected_calls = (
        [(1, 6)] * n_real
        if method == TopkTransformMethod.RAGGED
        else [(2, 6), (2, 6), (1, 6)]
    )
    assert calls == expected_calls
    assert torch.equal(result[:n_real, 0], pool_lens)
    assert torch.all(result[n_real:] == -1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires one GPU")
def test_grouped_chunking_rejects_request_row_larger_than_budget():
    device = torch.device("cuda")
    k_rows = 10
    budget = 4
    plan = SimpleNamespace(
        seq_lens_expanded=torch.tensor([k_rows], dtype=torch.int32, device=device),
        pooled_seq_lens_expanded=torch.tensor(
            [k_rows], dtype=torch.int32, device=device
        ),
        ragged_q_ks=torch.tensor([0], dtype=torch.int32, device=device),
        ragged_q_ke=torch.tensor([k_rows], dtype=torch.int32, device=device),
        ragged_groups=(
            SimpleNamespace(q_start=0, q_len=1, k_start=0, k_rows=k_rows),
        ),
    )
    q = torch.zeros((1, 1, 8), dtype=torch.float8_e4m3fn, device=device)
    k = torch.zeros((k_rows, 8), dtype=torch.float8_e4m3fn, device=device)
    scale = torch.ones(k_rows, dtype=torch.float32, device=device)
    weights = torch.ones((1, 1), dtype=torch.float32, device=device)

    with (
        mock.patch(
            "sglang.srt.layers.attention.dsa.dsa_indexer_kpool.deep_gemm",
            SimpleNamespace(fp8_mqa_logits=mock.Mock()),
            create=True,
        ) as deep_gemm_mock,
        pytest.raises(RuntimeError, match="required=40 bytes, budget=4 bytes"),
    ):
        _indexer()._topk_ragged_kpool_grouped(
            plan=plan,
            q_fp8=q,
            weights=weights,
            k_fp8=k,
            k_scale=scale,
            logits_budget_bytes=budget,
            total_q=1,
            page_table=None,
            page_table_row_index=None,
            topk_offsets=None,
        )

    deep_gemm_mock.fp8_mqa_logits.assert_not_called()
