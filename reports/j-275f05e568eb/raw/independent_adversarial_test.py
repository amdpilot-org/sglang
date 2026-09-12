import asyncio
import unittest
from types import SimpleNamespace

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.io_struct import (
    UpdateWeightFromDiskReqInput,
    UpdateWeightFromDiskReqOutput,
)
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.utils.aio_rwlock import RWLock


class Boundary:
    def __init__(self, manager):
        self.manager = manager
        self.requests = []

    def dispatch(self, request):
        self.requests.append(request)

    async def wait_count(self, count):
        async def wait():
            while len(self.requests) < count:
                await asyncio.sleep(0)

        await asyncio.wait_for(wait(), 1)

    def complete(self, message, *, success=True, paused=0):
        self.manager._handle_update_weights_from_disk_req_output(
            UpdateWeightFromDiskReqOutput(
                success=success,
                message=message,
                num_paused_requests=paused,
            )
        )


def manager(*, workers=1):
    value = TokenizerManager.__new__(TokenizerManager)
    value.server_args = SimpleNamespace(
        checkpoint_engine_wait_weights_before_ready=False
    )
    value.elastic_worker_count = workers
    value.event_loop = asyncio.get_running_loop()
    value.asyncio_tasks = set()
    value.mm_processor = None
    value.model_update_lock = RWLock()
    value.model_update_operation_lock = asyncio.Lock()
    value.model_update_result = None
    value.model_update_expected_workers = workers
    value.model_update_tmp = []
    value.is_pause_cond = asyncio.Condition()
    value.is_pause = True
    value._update_model_path_info = lambda *_: None
    boundary = Boundary(value)
    value._dispatch_to_scheduler = boundary.dispatch
    return value, boundary


def request(name):
    return UpdateWeightFromDiskReqInput(model_path=name, load_format="dummy")


class IndependentAdversarialTests(unittest.IsolatedAsyncioTestCase):
    async def test_three_overlapping_calls_keep_fifo_response_ownership(self):
        tm, scheduler = manager()
        tasks = [
            asyncio.create_task(tm.update_weights_from_disk(request(str(i))))
            for i in range(3)
        ]
        for i in range(3):
            await scheduler.wait_count(i + 1)
            self.assertEqual(len(scheduler.requests), i + 1)
            scheduler.complete(f"reply-{i}", paused=i)
            self.assertEqual(
                await asyncio.wait_for(tasks[i], 1),
                (True, f"reply-{i}", i),
            )
        self.assertEqual([r.model_path for r in scheduler.requests], ["0", "1", "2"])

    async def test_cancelled_queued_call_never_steals_or_dispatches(self):
        tm, scheduler = manager()
        first = asyncio.create_task(tm.update_weights_from_disk(request("first")))
        await scheduler.wait_count(1)
        queued = asyncio.create_task(tm.update_weights_from_disk(request("queued")))
        await asyncio.sleep(0)
        queued.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await queued
        third = asyncio.create_task(tm.update_weights_from_disk(request("third")))
        scheduler.complete("first reply")
        self.assertEqual(await first, (True, "first reply", 0))
        await scheduler.wait_count(2)
        self.assertEqual([r.model_path for r in scheduler.requests], ["first", "third"])
        scheduler.complete("third reply")
        self.assertEqual(await third, (True, "third reply", 0))

    async def test_cancelled_active_failure_is_drained_before_successor(self):
        tm, scheduler = manager()
        first = asyncio.create_task(tm.update_weights_from_disk(request("first")))
        await scheduler.wait_count(1)
        first.cancel()
        second = asyncio.create_task(tm.update_weights_from_disk(request("second")))
        await asyncio.sleep(0)
        self.assertEqual(len(scheduler.requests), 1)
        scheduler.complete("first failed", success=False)
        with self.assertRaises(asyncio.CancelledError):
            await first
        await scheduler.wait_count(2)
        scheduler.complete("second ok")
        self.assertEqual(await second, (True, "second ok", 0))

    async def test_concurrent_multiworker_responses_do_not_cross_operations(self):
        tm, scheduler = manager(workers=2)
        first = asyncio.create_task(tm.update_weights_from_disk(request("first")))
        second = asyncio.create_task(tm.update_weights_from_disk(request("second")))
        await scheduler.wait_count(1)
        scheduler.complete("first-worker-0", paused=3)
        await asyncio.sleep(0)
        self.assertFalse(first.done())
        self.assertEqual(len(scheduler.requests), 1)
        scheduler.complete("first-worker-1", paused=4)
        self.assertEqual(
            await first,
            (True, "first-worker-0 | first-worker-1", [3, 4]),
        )
        await scheduler.wait_count(2)
        scheduler.complete("second-worker-0", paused=5)
        scheduler.complete("second-worker-1", paused=6)
        self.assertEqual(
            await second,
            (True, "second-worker-0 | second-worker-1", [5, 6]),
        )


if __name__ == "__main__":
    unittest.main()
