import logging

import pytest
import torch

from sglang.kernels.ops.attention.minimax_sparse.prefill.flash_with_topk_idx import (
    flash_prefill_with_topk_index,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=30, suite="nightly-amd-kernel-1-gpu", nightly=True)

BLOCK_SIZE_Q = 64
BLOCK_SIZE_K = 128
TOPK = 16


def _build_inputs(seq_values):
    torch.manual_seed(31133)
    total_q = sum(seq_values)
    max_len = max(seq_values)
    q = torch.randn(total_q, 1, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(total_q, 1, 128, device="cuda", dtype=torch.bfloat16)
    req_to_token = torch.empty(
        len(seq_values), max_len, device="cuda", dtype=torch.int32
    )
    cu_values = [0]
    for batch_idx, seq_len in enumerate(seq_values):
        start = cu_values[-1]
        req_to_token[batch_idx] = torch.arange(
            start, start + max_len, device="cuda", dtype=torch.int32
        )
        cu_values.append(start + seq_len)
    return {
        "q": q,
        "k_cache": k,
        "v_cache": None,
        "sink": None,
        "req_to_token": req_to_token,
        "slot_ids": torch.arange(len(seq_values), device="cuda", dtype=torch.int64),
        "cu_seqlens": torch.tensor(cu_values, device="cuda", dtype=torch.int32),
        "seq_lens": torch.tensor(seq_values, device="cuda", dtype=torch.int32),
        "prefix_lens": torch.zeros(len(seq_values), device="cuda", dtype=torch.int32),
        "max_seqlen_q": max_len,
        "block_size_q": BLOCK_SIZE_Q,
        "block_size_k": BLOCK_SIZE_K,
        "topk": TOPK,
        "init_blocks": 1,
        "local_blocks": 2,
        "disable_index_value": True,
    }


def _run(inputs, max_seqlen_k):
    _, topk_idx = flash_prefill_with_topk_index(**inputs, max_seqlen_k=max_seqlen_k)
    torch.cuda.synchronize()
    return topk_idx.cpu()


def _torch_topk_reference(inputs):
    q = inputs["q"].float().cpu()[:, 0]
    k = inputs["k_cache"].float().cpu()[:, 0]
    seq_lens = inputs["seq_lens"].cpu().tolist()
    cu = inputs["cu_seqlens"].cpu().tolist()
    rows = []
    scale = q.shape[-1] ** -0.5
    for batch_idx, seq_len in enumerate(seq_lens):
        start = cu[batch_idx]
        for query_offset in range(0, seq_len, BLOCK_SIZE_Q):
            valid_blocks = (query_offset + BLOCK_SIZE_K) // BLOCK_SIZE_K
            scores = []
            for block_idx in range(valid_blocks):
                block_start = block_idx * BLOCK_SIZE_K
                block_end = min(block_start + BLOCK_SIZE_K, query_offset + 1)
                if block_end <= block_start:
                    score = float("-inf")
                else:
                    score = float(
                        torch.mv(
                            k[start + block_start : start + block_end],
                            q[start + query_offset],
                        ).max()
                        * scale
                    )
                scores.append(score)
            scores[0] = 1e30
            local_start = max(0, valid_blocks - 2)
            for block_idx in range(local_start, valid_blocks):
                scores[block_idx] = 1e29
            actual_topk = min(TOPK, valid_blocks)
            selected = torch.tensor(scores).topk(actual_topk).indices.sort().values
            row = torch.full((TOPK,), -1, dtype=torch.int32)
            row[:actual_topk] = selected.to(torch.int32)
            rows.append(row)
    return torch.stack(rows).unsqueeze(0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_stale_max_seqlen_k_matches_live_width_and_torch_reference(caplog):
    inputs = _build_inputs([64, 4096])
    reference = _run(inputs, 4096)
    torch.testing.assert_close(reference, _torch_topk_reference(inputs), rtol=0, atol=0)

    with caplog.at_level(logging.WARNING):
        stale = _run(inputs, 64)

    torch.testing.assert_close(stale, reference, rtol=0, atol=0)
    assert "enlarging score buffer from 1 to 32 block-columns" in caplog.text


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize("caller_max", [4096, 4095])
def test_score_width_boundary_does_not_enlarge(caller_max, caplog):
    inputs = _build_inputs([4096])
    with caplog.at_level(logging.WARNING):
        result = _run(inputs, caller_max)

    assert result.shape == (1, 64, TOPK)
    assert "enlarging score buffer" not in caplog.text
