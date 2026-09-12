import asyncio
import threading
from contextvars import ContextVar

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.tokenizer_manager import TokenizerManager


async def main():
    manager = TokenizerManager.__new__(TokenizerManager)
    manager.init_request_preprocessor()
    marker = ContextVar("marker")
    active = 0
    max_active = 0
    lock = threading.Lock()

    def work(expected):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        try:
            assert marker.get() == expected
            return expected
        finally:
            with lock:
                active -= 1

    async def submit(value):
        token = marker.set(value)
        try:
            return await manager.run_in_request_preprocessor(work, value)
        finally:
            marker.reset(token)

    try:
        values = [f"request-{i}" for i in range(50)]
        assert await asyncio.gather(*(submit(v) for v in values)) == values
        assert max_active == 1

        try:
            await manager.run_in_request_preprocessor(
                lambda: (_ for _ in ()).throw(ValueError("sentinel"))
            )
        except ValueError as exc:
            assert str(exc) == "sentinel"
        else:
            raise AssertionError("worker exception was not propagated")

        assert await manager.run_in_request_preprocessor(lambda: 42) == 42
        print("PASS: 50 contexts isolated, max_active=1, exception propagated, executor recovered")
    finally:
        manager._request_preprocessor_executor.shutdown(wait=True)


asyncio.run(main())
