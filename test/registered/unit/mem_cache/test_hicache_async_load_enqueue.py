import threading
import time
import unittest
from queue import Queue
from unittest.mock import MagicMock, Mock, patch

import torch

from sglang.srt.managers.cache_controller import (
    CacheOperation,
    HiCacheController,
    LayerDoneCounter,
    LayerLoadingEvent,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


class FakeEvent:
    def __init__(self, *args, **kwargs):
        self.recorded = False

    def record(self, stream=None):
        self.recorded = True

    def query(self):
        return self.recorded


class FakeStream:
    def __init__(self):
        self.waited = []

    def wait_event(self, event):
        self.waited.append(event)


class TestLayerLoadingEvent(unittest.TestCase):
    def test_wait_blocks_until_device_event_has_been_recorded(self):
        stream = FakeStream()
        with (
            patch(
                "sglang.srt.managers.cache_controller.device_module.Event", FakeEvent
            ),
            patch(
                "sglang.srt.managers.cache_controller.device_module.current_stream",
                return_value=stream,
            ),
        ):
            event = LayerLoadingEvent(1)
            event.begin_enqueue()
            finished = threading.Event()
            waiter = threading.Thread(target=lambda: (event.wait(0), finished.set()))
            waiter.start()
            time.sleep(0.02)
            self.assertFalse(finished.is_set())
            self.assertEqual(stream.waited, [])

            event.complete(0)
            waiter.join(timeout=1)
            self.assertTrue(finished.is_set())
            self.assertEqual(stream.waited, [event.load_events[0]])

    def test_async_start_hands_off_without_moving_indices(self):
        controller = Mock(spec=HiCacheController)
        op = CacheOperation(torch.arange(2), torch.arange(2), 7)
        controller.load_queue = [op]
        controller.layer_done_counter = MagicMock()
        controller.layer_done_counter.update_producer.return_value = 1
        producer_event = controller.layer_done_counter.events[1]
        controller.load_fence_stream = object()
        controller.async_load_enqueue = True
        controller.load_enqueue_queue = Queue()

        with patch(
            "sglang.srt.managers.cache_controller.device_module.Event", FakeEvent
        ):
            self.assertEqual(HiCacheController.start_loading(controller), 1)

        task = controller.load_enqueue_queue.get_nowait()
        self.assertIs(task[1], producer_event)
        self.assertEqual(task[2].node_ids, [7])
        self.assertTrue(task[3].recorded)
        producer_event.start_event.record.assert_called_once_with()
        controller._move_op_indices.assert_not_called()

    def test_slot_rotation_waits_for_enqueue_completion(self):
        with patch(
            "sglang.srt.managers.cache_controller.device_module.Event", FakeEvent
        ):
            counter = LayerDoneCounter(1)
            for event in counter.events:
                event.load_events[-1].record()

            first = counter.update_producer()
            counter.events[first].complete(0)
            counter.update_producer()
            counter.events[1].complete(0)
            counter.events[1].mark_enqueue_done()
            counter.update_producer()
            counter.events[2].complete(0)
            counter.events[2].mark_enqueue_done()

            finished = threading.Event()
            waiter = threading.Thread(
                target=lambda: (counter.update_producer(), finished.set())
            )
            waiter.start()
            time.sleep(0.02)
            self.assertFalse(finished.is_set())
            counter.events[first].mark_enqueue_done()
            waiter.join(timeout=1)
            self.assertTrue(finished.is_set())


if __name__ == "__main__":
    unittest.main()
