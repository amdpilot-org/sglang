from unittest.mock import Mock

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
