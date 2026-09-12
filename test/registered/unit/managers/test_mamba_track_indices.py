from types import SimpleNamespace

import pytest
import torch

from sglang.srt.managers.schedule_batch import set_mamba_track_indices_from_reqs
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
from sglang.srt.runtime_context import get_context
from sglang.srt.speculative.spec_utils import prepare_mamba_track_for_verify


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a GPU-backed mamba buffer"
)


def _batch(track_positions):
    buffers = torch.tensor(
        [[101, 102], [201, 202], [301, 302]],
        dtype=torch.int64,
        device="cuda",
    )
    reqs = [
        SimpleNamespace(kv=SimpleNamespace(mamba_next_track_idx=position))
        for position in track_positions
    ]
    return SimpleNamespace(
        req_to_token_pool=SimpleNamespace(
            req_index_to_mamba_ping_pong_track_buffer_mapping=buffers
        ),
        req_pool_indices=torch.arange(len(reqs), device="cuda"),
        reqs=reqs,
    )


def test_fresh_request_defaults_to_first_mamba_track_slot():
    batch = _batch([None])

    set_mamba_track_indices_from_reqs(batch)

    assert batch.mamba_track_buffer_indices == [0]
    assert batch.mamba_track_indices.cpu().tolist() == [101]
    # Building verify metadata must not claim that request allocation occurred.
    assert batch.reqs[0].kv.mamba_next_track_idx is None


def test_mamba_track_indices_keep_initialized_slots_in_mixed_batch():
    batch = _batch([None, 1, 0])

    set_mamba_track_indices_from_reqs(batch)

    assert batch.mamba_track_buffer_indices == [0, 1, 0]
    assert batch.mamba_track_indices.cpu().tolist() == [101, 202, 301]


def test_lazy_spec_track_plan_overrides_request_slots():
    batch = _batch([0, 1, None])

    set_mamba_track_indices_from_reqs(batch, track_positions=[1, 0, 1])

    assert batch.mamba_track_buffer_indices == [1, 0, 1]
    assert batch.mamba_track_indices.cpu().tolist() == [102, 201, 302]


def test_lazy_spec_fresh_allocation_verify_and_boundary_promotion():
    """Exercise the lifecycle around the helper, not just index selection."""
    device = "cuda"
    allocated = iter(
        [
            torch.tensor([41], dtype=torch.int64, device=device),
            torch.tensor([42], dtype=torch.int64, device=device),
        ]
    )
    freed = []
    allocator = SimpleNamespace(
        alloc=lambda count: next(allocated),
        free=lambda slots: freed.append(slots.clone()),
    )
    pool = SimpleNamespace(
        enable_mamba_extra_buffer_lazy=True,
        mamba_ping_pong_track_buffer_size=2,
        mamba_allocator=allocator,
        req_index_to_mamba_ping_pong_track_buffer_mapping=torch.full(
            (4, 2), -1, dtype=torch.int64, device=device
        ),
    )
    pool.set_mamba_ping_pong_slot = (
        lambda req, idx, value: HybridReqToTokenPool.set_mamba_ping_pong_slot(
            pool, req, idx, value
        )
    )
    req = SimpleNamespace(
        rid="fresh",
        finished=lambda: False,
        kv=SimpleNamespace(
            req_pool_idx=2,
            kv_committed_len=2,
            mamba_ping_pong_track_buffer=None,
            mamba_next_track_idx=None,
            mamba_last_track_idx=None,
            mamba_last_track_seqlen=None,
        ),
    )

    # This is the real fresh-request allocator used by HybridReqToTokenPool.
    HybridReqToTokenPool._alloc_ping_pong_buffer(pool, req)
    pool.req_index_to_mamba_ping_pong_track_buffer_mapping[2] = (
        req.kv.mamba_ping_pong_track_buffer
    )
    assert req.kv.mamba_next_track_idx == 0
    assert req.kv.mamba_ping_pong_track_buffer.cpu().tolist() == [41, -1]

    batch = SimpleNamespace(
        reqs=[req],
        req_to_token_pool=pool,
        req_pool_indices=torch.tensor([2], dtype=torch.int64, device=device),
        tree_cache=SimpleNamespace(page_size=4),
        mamba_lazy_spec_track_positions_cpu=None,
        mamba_track_mask=torch.tensor([True], device=device),
        mamba_track_seqlens=torch.tensor([2], device=device),
    )
    with get_context().override_server_args(
        mamba_radix_cache_strategy="extra_buffer_lazy"
    ):
        # A verify window from committed length 2 can cross the boundary at 4.
        from sglang.srt.managers.schedule_batch import ScheduleBatch

        ScheduleBatch.mamba_lazy_spec_prepare(
            batch, mamba_track_interval=4, max_draft_tokens=1
        )
        prepare_mamba_track_for_verify(batch)

    assert batch.mamba_lazy_spec_track_positions_cpu == [1]
    assert req.kv.mamba_ping_pong_track_buffer.cpu().tolist() == [41, 42]
    assert batch.mamba_track_indices.cpu().tolist() == [42]
    assert batch.mamba_track_mask is None
    assert batch.mamba_track_seqlens is None

    processor = SchedulerBatchResultProcessor.__new__(
        SchedulerBatchResultProcessor
    )
    processor._mamba_lazy_spec_update(
        req, batch, i=0, crossed=True, track_seqlen=4
    )

    assert [slots.cpu().tolist() for slots in freed] == [[41]]
    assert req.kv.mamba_ping_pong_track_buffer.cpu().tolist() == [-1, 42]
    assert req.kv.mamba_next_track_idx == 1
    assert req.kv.mamba_last_track_idx == 1
    assert req.kv.mamba_last_track_seqlen == 4
