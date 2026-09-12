import pickle
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.entrypoints.control_requests import ShutdownReq
from sglang.multimodal_gen.runtime.managers.scheduler import (
    Scheduler,
    _is_unrecoverable_accelerator_error,
)


def test_device_not_ready_is_unrecoverable():
    assert _is_unrecoverable_accelerator_error(
        "RuntimeError: CUDA driver error: device not ready"
    )


def test_caching_allocator_invariant_failure_is_unrecoverable():
    assert _is_unrecoverable_accelerator_error(
        '!handles_.at(i) INTERNAL ASSERT FAILED at "CUDACachingAllocator.cpp":467'
    )


def test_ordinary_oom_is_not_misclassified_as_poisoned_device():
    assert not _is_unrecoverable_accelerator_error("CUDA error: out of memory")


def test_unrelated_execution_error_is_not_misclassified():
    assert not _is_unrecoverable_accelerator_error("shape mismatch in attention")


def test_unrecoverable_error_latches_fail_fast_state_after_actionable_reply():
    scheduler = Scheduler.__new__(Scheduler)
    scheduler._running = True
    scheduler._fatal_error_message = None

    result = scheduler._handle_execution_error(
        RuntimeError("CUDA driver error: device not ready")
    )

    assert scheduler._running is True
    assert "restart the server" in result.error
    assert scheduler._fatal_error_message == result.error


def test_request_after_unrecoverable_error_is_rejected_without_worker_dispatch():
    scheduler = Scheduler.__new__(Scheduler)
    scheduler._fatal_error_message = "accelerator unavailable; restart the server"
    scheduler.request_handlers = {object: Mock()}

    result = scheduler._dispatch_single_request(object())

    assert result.error == scheduler._fatal_error_message
    scheduler.request_handlers[object].assert_not_called()


def test_shutdown_is_still_dispatched_after_unrecoverable_error():
    scheduler = Scheduler.__new__(Scheduler)
    scheduler._fatal_error_message = "accelerator unavailable; restart the server"
    expected = object()
    handler = Mock(return_value=expected)
    scheduler.request_handlers = {ShutdownReq: handler}
    request = ShutdownReq()

    assert scheduler._dispatch_single_request(request) is expected
    handler.assert_called_once_with([request])


def test_recoverable_error_keeps_scheduler_running():
    scheduler = Scheduler.__new__(Scheduler)
    scheduler._running = True
    scheduler._fatal_error_message = None

    result = scheduler._handle_execution_error(RuntimeError("CUDA error: out of memory"))

    assert scheduler._running is True
    assert scheduler._fatal_error_message is None
    assert result.error == "CUDA error: out of memory"


def test_disagg_encoder_rejects_later_work_after_unrecoverable_error(monkeypatch):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler._running = True
    scheduler._fatal_error_message = None
    scheduler._disagg_role = RoleType.ENCODER
    scheduler.gpu_id = 0
    scheduler.server_args = SimpleNamespace(
        sp_degree=1, tp_size=1, enable_cfg_parallel=False
    )
    scheduler._compute_ready_queue = None
    scheduler._consecutive_error_count = 0
    scheduler._max_consecutive_errors = 3
    scheduler._pool_result_push = object()
    scheduler._disagg_metrics = None
    scheduler._cleanup_disagg = Mock()

    frames = [pickle.dumps(SimpleNamespace(request_id="request-1"))]
    scheduler._disagg_recv_work = Mock(side_effect=[frames, frames])
    dispatch = Mock(side_effect=RuntimeError("CUDA driver error: device not ready"))
    scheduler._disagg_encoder_step = dispatch
    replies = []

    def capture_reply(_socket, _tensors, scalar_fields):
        replies.append(scalar_fields)
        if len(replies) == 2:
            scheduler._running = False

    monkeypatch.setattr(
        "sglang.multimodal_gen.runtime.disaggregation.scheduler_mixin.send_tensors",
        capture_reply,
    )

    scheduler._disagg_event_loop()

    dispatch.assert_called_once()
    assert len(replies) == 2
    assert all(reply["request_id"] == "request-1" for reply in replies)
    assert all("restart the server" in reply["_disagg_error"] for reply in replies)
