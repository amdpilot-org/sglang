import unittest
from types import SimpleNamespace

from sglang.srt.managers.scheduler_components.metrics_reporter import (
    _average_request_queue_latency,
)


def _req(wait_queue_entry_time):
    return SimpleNamespace(
        time_stats=SimpleNamespace(
            wait_queue_entry_time=wait_queue_entry_time,
        )
    )


class TestAverageRequestQueueLatency(unittest.TestCase):
    def test_averages_current_waiting_requests(self):
        self.assertEqual(
            _average_request_queue_latency([_req(91.0), _req(97.0)], now=101.0),
            7.0,
        )

    def test_empty_queue_is_zero(self):
        self.assertEqual(_average_request_queue_latency([], now=101.0), 0.0)

    def test_uninitialized_entry_time_is_excluded(self):
        self.assertEqual(
            _average_request_queue_latency([_req(0.0), _req(96.0)], now=101.0),
            5.0,
        )

    def test_future_entry_time_is_clamped(self):
        self.assertEqual(
            _average_request_queue_latency([_req(102.0)], now=101.0),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
