"""Behavioral regression probe for fatal scheduler state handling."""

from unittest.mock import Mock

from sglang.multimodal_gen.runtime.managers.scheduler import Scheduler


scheduler = Scheduler.__new__(Scheduler)
scheduler._running = True
scheduler._fatal_error_message = None
scheduler.request_handlers = {object: Mock()}

first = scheduler._handle_execution_error(
    RuntimeError("CUDA driver error: device not ready")
)
assert scheduler._running is True
assert "restart the server" in first.error

second = scheduler._dispatch_single_request(object())
assert second.error == first.error
scheduler.request_handlers[object].assert_not_called()

print("fatal request replied; later request rejected without worker dispatch")
