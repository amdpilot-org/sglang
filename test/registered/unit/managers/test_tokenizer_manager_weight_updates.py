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
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _SchedulerBoundary:
    def __init__(self, manager: TokenizerManager):
        self.manager = manager
        self.requests = []

    def dispatch(self, request: UpdateWeightFromDiskReqInput):
        self.requests.append(request)

    async def wait_for_requests(self, count: int):
        async def wait():
            while len(self.requests) < count:
                await asyncio.sleep(0)

        await asyncio.wait_for(wait(), timeout=1)

    def complete(self, message: str):
        self.manager._handle_update_weights_from_disk_req_output(
            UpdateWeightFromDiskReqOutput(success=True, message=message)
        )


def _manager(*, paused: bool = True, workers: int = 1):
    manager = TokenizerManager.__new__(TokenizerManager)
    manager.server_args = SimpleNamespace(
        checkpoint_engine_wait_weights_before_ready=False,
    )
    manager.elastic_worker_count = workers
    manager.event_loop = asyncio.get_running_loop()
    manager.asyncio_tasks = set()
    manager.mm_processor = None
    manager.model_update_lock = RWLock()
    manager.model_update_operation_lock = asyncio.Lock()
    manager.model_update_result = None
    manager.model_update_expected_workers = workers
    manager.model_update_tmp = []
    manager.is_pause_cond = asyncio.Condition()
    manager.is_pause = paused
    manager._update_model_path_info = lambda *_: None

    scheduler = _SchedulerBoundary(manager)
    manager._dispatch_to_scheduler = scheduler.dispatch
    return manager, scheduler


def _request(path: str):
    return UpdateWeightFromDiskReqInput(model_path=path, load_format="dummy")


class TestTokenizerManagerWeightUpdates(unittest.IsolatedAsyncioTestCase):
    async def test_paused_update_blocks_resume_until_completion(self):
        manager, scheduler = _manager()
        update = asyncio.create_task(manager.update_weights_from_disk(_request("model")))
        await scheduler.wait_for_requests(1)

        async def resume():
            async with manager.is_pause_cond:
                manager.is_pause = False

        resume_task = asyncio.create_task(resume())
        await asyncio.sleep(0)
        self.assertFalse(resume_task.done())

        scheduler.complete("complete")
        self.assertEqual(
            await asyncio.wait_for(update, timeout=1),
            (True, "complete", 0),
        )
        await asyncio.wait_for(resume_task, timeout=1)

    async def test_paused_concurrent_updates_each_receive_own_completion(self):
        manager, scheduler = _manager()
        first = asyncio.create_task(manager.update_weights_from_disk(_request("first")))
        second = asyncio.create_task(manager.update_weights_from_disk(_request("second")))

        try:
            await scheduler.wait_for_requests(1)
            scheduler.complete("first complete")

            done, _ = await asyncio.wait({first}, timeout=0.1)
            self.assertIn(first, done, "the first update lost its completion")
            self.assertEqual(first.result(), (True, "first complete", 0))

            await scheduler.wait_for_requests(2)
            scheduler.complete("second complete")
            self.assertEqual(
                await asyncio.wait_for(second, timeout=1),
                (True, "second complete", 0),
            )
        finally:
            for task in (first, second):
                if not task.done():
                    task.cancel()
            await asyncio.gather(first, second, return_exceptions=True)

    async def test_cancellation_keeps_ownership_until_response_is_drained(self):
        manager, scheduler = _manager()
        first = asyncio.create_task(manager.update_weights_from_disk(_request("first")))
        await scheduler.wait_for_requests(1)
        first.cancel()
        await asyncio.sleep(0)

        second = asyncio.create_task(manager.update_weights_from_disk(_request("second")))
        try:
            await asyncio.sleep(0)
            self.assertEqual(
                len(scheduler.requests),
                1,
                "a cancelled update released ownership before its response arrived",
            )

            scheduler.complete("first complete")
            with self.assertRaises(asyncio.CancelledError):
                await first

            await scheduler.wait_for_requests(2)
            scheduler.complete("second complete")
            self.assertEqual(
                await asyncio.wait_for(second, timeout=1),
                (True, "second complete", 0),
            )
        finally:
            for task in (first, second):
                if not task.done():
                    task.cancel()
            await asyncio.gather(first, second, return_exceptions=True)

    async def test_repeated_cancellation_cannot_interrupt_response_drain(self):
        manager, scheduler = _manager()
        update = asyncio.create_task(manager.update_weights_from_disk(_request("model")))
        await scheduler.wait_for_requests(1)

        update.cancel()
        await asyncio.sleep(0)
        update.cancel()
        await asyncio.sleep(0)
        self.assertFalse(
            update.done(),
            "repeated cancellation released ownership before the response",
        )

        scheduler.complete("complete")
        with self.assertRaises(asyncio.CancelledError):
            await update

    async def test_cancelled_successful_update_still_publishes_weight_version(self):
        manager, scheduler = _manager()
        published_versions = []
        manager._update_weight_version_if_provided = published_versions.append
        request = UpdateWeightFromDiskReqInput(
            model_path="model",
            load_format="dummy",
            weight_version="version-1",
        )
        update = asyncio.create_task(manager.update_weights_from_disk(request))
        await scheduler.wait_for_requests(1)

        update.cancel()
        await asyncio.sleep(0)
        scheduler.complete("complete")

        with self.assertRaises(asyncio.CancelledError):
            await update
        self.assertEqual(published_versions, ["version-1"])

    async def test_unpaused_updates_preserve_writer_preference(self):
        manager, scheduler = _manager(paused=False)
        await manager.model_update_lock.acquire_reader()

        first = asyncio.create_task(manager.update_weights_from_disk(_request("first")))
        while manager.model_update_lock._waiting_writers < 1:
            await asyncio.sleep(0)
        second = asyncio.create_task(manager.update_weights_from_disk(_request("second")))
        while manager.model_update_lock._waiting_writers < 2:
            await asyncio.sleep(0)

        late_reader_acquired = asyncio.Event()
        release_late_reader = asyncio.Event()

        async def late_reader():
            async with manager.model_update_lock.reader_lock:
                late_reader_acquired.set()
                await release_late_reader.wait()

        reader_task = asyncio.create_task(late_reader())
        await manager.model_update_lock.release_reader()

        await scheduler.wait_for_requests(1)
        scheduler.complete("first complete")
        second_dispatched = asyncio.create_task(scheduler.wait_for_requests(2))
        reader_arrived = asyncio.create_task(late_reader_acquired.wait())
        done, pending = await asyncio.wait(
            {second_dispatched, reader_arrived},
            return_when=asyncio.FIRST_COMPLETED,
            timeout=1,
        )
        second_kept_priority = second_dispatched in done

        if second_kept_priority:
            scheduler.complete("second complete")
        else:
            release_late_reader.set()
            await scheduler.wait_for_requests(2)
            scheduler.complete("second complete")

        await asyncio.wait_for(asyncio.gather(first, second), timeout=1)
        release_late_reader.set()
        await asyncio.wait_for(reader_task, timeout=1)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)

        self.assertTrue(
            second_kept_priority,
            "a later generation reader overtook an already queued update",
        )

    async def test_sequential_paused_updates_are_unchanged(self):
        manager, scheduler = _manager()

        for index in range(2):
            update = asyncio.create_task(
                manager.update_weights_from_disk(_request(f"model-{index}"))
            )
            await scheduler.wait_for_requests(index + 1)
            scheduler.complete(f"complete-{index}")
            self.assertEqual(
                await asyncio.wait_for(update, timeout=1),
                (True, f"complete-{index}", 0),
            )

    async def test_multi_worker_update_waits_for_all_worker_responses(self):
        manager, scheduler = _manager(workers=2)
        update = asyncio.create_task(manager.update_weights_from_disk(_request("model")))
        await scheduler.wait_for_requests(1)

        scheduler.complete("worker-0")
        await asyncio.sleep(0)
        self.assertFalse(update.done())

        scheduler.complete("worker-1")
        self.assertEqual(
            await asyncio.wait_for(update, timeout=1),
            (True, "worker-0 | worker-1", [0, 0]),
        )


if __name__ == "__main__":
    unittest.main()
