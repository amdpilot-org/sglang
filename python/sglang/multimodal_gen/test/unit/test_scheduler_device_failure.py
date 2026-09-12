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


def test_unrecoverable_error_stops_scheduler_after_actionable_reply():
    scheduler = Scheduler.__new__(Scheduler)
    scheduler._running = True

    result = scheduler._handle_execution_error(
        RuntimeError("CUDA driver error: device not ready")
    )

    assert scheduler._running is False
    assert "restart the server" in result.error


def test_recoverable_error_keeps_scheduler_running():
    scheduler = Scheduler.__new__(Scheduler)
    scheduler._running = True

    result = scheduler._handle_execution_error(RuntimeError("CUDA error: out of memory"))

    assert scheduler._running is True
    assert result.error == "CUDA error: out of memory"
