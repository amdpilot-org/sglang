from types import SimpleNamespace

import torch

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
from sglang.srt.runtime_context import get_context
from sglang.srt.speculative.spec_utils import prepare_mamba_track_for_verify


def make_case(*, committed_len, allocations, initial=(41, -1)):
    device = "cuda"
    allocated = iter(allocations)
    calls = []

    def alloc(count):
        calls.append(count)
        return next(allocated)

    pool = SimpleNamespace(
        mamba_allocator=SimpleNamespace(alloc=alloc),
        req_index_to_mamba_ping_pong_track_buffer_mapping=torch.full(
            (1, 2), -1, dtype=torch.int64, device=device
        ),
    )
    pool.set_mamba_ping_pong_slot = (
        lambda req, idx, value: HybridReqToTokenPool.set_mamba_ping_pong_slot(
            pool, req, idx, value
        )
    )
    req = SimpleNamespace(
        kv=SimpleNamespace(
            req_pool_idx=0,
            kv_committed_len=committed_len,
            mamba_next_track_idx=0,
            mamba_ping_pong_track_buffer=torch.tensor(
                initial, dtype=torch.int64, device=device
            ),
        )
    )
    pool.req_index_to_mamba_ping_pong_track_buffer_mapping[0] = (
        req.kv.mamba_ping_pong_track_buffer
    )
    batch = SimpleNamespace(
        reqs=[req],
        req_to_token_pool=pool,
        req_pool_indices=torch.tensor([0], dtype=torch.int64, device=device),
        mamba_lazy_spec_track_positions_cpu=None,
        mamba_track_mask=torch.tensor([True], device=device),
        mamba_track_seqlens=torch.tensor([committed_len], device=device),
    )
    return req, batch, calls


def test_no_reachable_boundary_does_not_allocate_or_redirect():
    req, batch, calls = make_case(committed_len=1, allocations=[])
    with get_context().override_server_args(
        mamba_radix_cache_strategy="extra_buffer_lazy"
    ):
        ScheduleBatch.mamba_lazy_spec_prepare(batch, 4, 1)
        prepare_mamba_track_for_verify(batch)
    assert calls == []
    assert batch.mamba_lazy_spec_track_positions_cpu == [0]
    assert batch.mamba_track_indices.cpu().tolist() == [41]
    assert req.kv.mamba_ping_pong_track_buffer.cpu().tolist() == [41, -1]


def test_reachable_boundary_allocation_failure_falls_back_in_place():
    req, batch, calls = make_case(committed_len=2, allocations=[None])
    with get_context().override_server_args(
        mamba_radix_cache_strategy="extra_buffer_lazy"
    ):
        ScheduleBatch.mamba_lazy_spec_prepare(batch, 4, 1)
        prepare_mamba_track_for_verify(batch)
    assert calls == [1]
    assert batch.mamba_lazy_spec_track_positions_cpu == [0]
    assert batch.mamba_track_indices.cpu().tolist() == [41]
    assert req.kv.mamba_ping_pong_track_buffer.cpu().tolist() == [41, -1]


def test_existing_pending_slot_is_reused_without_allocation():
    req, batch, calls = make_case(committed_len=2, allocations=[], initial=(41, 77))
    with get_context().override_server_args(
        mamba_radix_cache_strategy="extra_buffer_lazy"
    ):
        ScheduleBatch.mamba_lazy_spec_prepare(batch, 4, 1)
        prepare_mamba_track_for_verify(batch)
    assert calls == []
    assert batch.mamba_lazy_spec_track_positions_cpu == [1]
    assert batch.mamba_track_indices.cpu().tolist() == [77]
    assert req.kv.mamba_ping_pong_track_buffer.cpu().tolist() == [41, 77]


def test_window_edge_is_inclusive_at_boundary():
    req, batch, calls = make_case(
        committed_len=6,
        allocations=[torch.tensor([88], dtype=torch.int64, device="cuda")],
    )
    with get_context().override_server_args(
        mamba_radix_cache_strategy="extra_buffer_lazy"
    ):
        ScheduleBatch.mamba_lazy_spec_prepare(batch, 8, 1)
        prepare_mamba_track_for_verify(batch)
    assert calls == [1]
    assert batch.mamba_lazy_spec_track_positions_cpu == [1]
    assert batch.mamba_track_indices.cpu().tolist() == [88]
