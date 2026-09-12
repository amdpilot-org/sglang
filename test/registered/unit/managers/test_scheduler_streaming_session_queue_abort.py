"""Regression tests for streaming-session requests removed before execution."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import AbortReq
from sglang.srt.managers.schedule_batch import FINISH_ABORT
from sglang.srt.managers.scheduler import Scheduler
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()


def _request(rid, session, *, holds_mamba=False):
    return SimpleNamespace(
        rid=rid,
        session=session,
        kv=SimpleNamespace(holds_kv=holds_mamba, holds_mamba=holds_mamba),
        output_ids=[],
        weight_version_events=[],
        finished_reason=None,
        beam_group=None,
        time_stats=SimpleNamespace(trace_ctx=MagicMock()),
    )


def _scheduler(waiting_queue):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.chunked_req = None
    scheduler.waiting_queue = waiting_queue
    scheduler.mm_receiver = None
    scheduler.enable_hierarchical_cache = False
    scheduler.enable_hicache_storage = False
    scheduler.enable_unified_cache_external_linker = False
    scheduler.tree_cache = MagicMock()
    scheduler.beam_coordinator = MagicMock()
    scheduler.ipc_channels = SimpleNamespace(send_to_tokenizer=MagicMock())
    scheduler.disaggregation_mode = DisaggregationMode.NULL
    scheduler.dllm_config = None
    scheduler.grammar_manager = MagicMock()
    scheduler.ps = SimpleNamespace(pp_size=1)
    scheduler.running_batch = None
    scheduler.last_batch = None
    return scheduler


@patch(
    "sglang.srt.managers.scheduler.get_serving",
    return_value=SimpleNamespace(weight_version="v0"),
)
def test_queued_abort_reopens_streaming_session_and_preserves_committed_turn(_):
    committed = object()
    session = SimpleNamespace(
        streaming=True, _inflight=True, req_nodes={"committed": committed}
    )
    session.abort_req = MagicMock(side_effect=lambda: setattr(session, "_inflight", False))
    aborted = _request("queued", session)
    unrelated = _request("other", None)
    scheduler = _scheduler([aborted, unrelated])

    scheduler.abort_request(AbortReq(rid="queued"))

    assert scheduler.waiting_queue == [unrelated]
    session.abort_req.assert_called_once_with()
    assert session._inflight is False
    assert session.req_nodes == {"committed": committed}


@patch(
    "sglang.srt.managers.scheduler.get_serving",
    return_value=SimpleNamespace(weight_version="v0"),
)
@patch("sglang.srt.managers.scheduler.release_kv_cache")
def test_queued_mamba_abort_is_marked_before_cache_release(release_kv_cache, _):
    session = SimpleNamespace(streaming=True, _inflight=True, abort_req=MagicMock())
    aborted = _request("queued-mamba", session, holds_mamba=True)
    scheduler = _scheduler([aborted])

    scheduler.abort_request(
        AbortReq(
            rid="queued-mamba",
            abort_message="Request waiting timeout reached.",
            finished_reason={
                "type": "abort",
                "status_code": 503,
                "message": "Request waiting timeout reached.",
            },
        )
    )

    release_kv_cache.assert_called_once_with(
        aborted, scheduler.tree_cache, is_insert=False
    )
    assert isinstance(aborted.finished_reason, FINISH_ABORT)
    assert aborted.finished_reason.message == "Request waiting timeout reached."
    assert aborted.finished_reason.status_code == 503
    session.abort_req.assert_called_once_with()


@patch(
    "sglang.srt.managers.scheduler.get_serving",
    return_value=SimpleNamespace(weight_version="v0"),
)
def test_queued_abort_does_not_touch_non_streaming_session(_):
    session = SimpleNamespace(streaming=False, abort_req=MagicMock())
    scheduler = _scheduler([_request("ordinary-session", session)])

    scheduler.abort_request(AbortReq(rid="ordinary-session"))

    session.abort_req.assert_not_called()
