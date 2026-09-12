import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.managers.scheduler_components.kv_events_publisher import (
    SchedulerKvEventsPublisher,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestSchedulerKvEventsPublisher(CustomTestCase):
    @patch(
        "sglang.srt.managers.scheduler_components.kv_events_publisher.sock_send"
    )
    def test_kv_block_metrics_use_page_units(self, mock_sock_send):
        cases = [
            ("paged_half_full", 8192, 16, 0.5, 512, 256),
            ("single_token_pages", 8192, 1, 0.5, 8192, 4096),
            ("empty_cache", 8192, 16, 0.0, 512, 0),
            ("partial_trailing_page", 8200, 16, 1.0, 512, 512),
        ]

        for label, max_tokens, page_size, usage, total_blocks, active_blocks in cases:
            with self.subTest(label=label):
                mock_sock_send.reset_mock()
                stats = SimpleNamespace(
                    num_running_reqs=SimpleNamespace(total=2),
                    num_queue_reqs=SimpleNamespace(total=3),
                    token_usage=usage,
                    cache_hit_rate=0.25,
                )
                socket = MagicMock()
                socket.closed = False
                publisher = SchedulerKvEventsPublisher(
                    kv_events_config=None,
                    ps=SimpleNamespace(dp_rank=None),
                    attn_tp_rank=0,
                    attn_cp_rank=0,
                    attn_dp_rank=0,
                    dp_rank=None,
                    tree_cache=MagicMock(),
                    send_metrics_from_scheduler=socket,
                    max_running_requests=8,
                    max_total_num_tokens=max_tokens,
                    page_size=page_size,
                    get_stats=lambda: stats,
                )
                publisher.enable_kv_cache_events = True

                publisher.emit_kv_metrics()

                metrics = mock_sock_send.call_args.args[1]
                self.assertEqual(metrics.kv_total_blocks, total_blocks)
                self.assertEqual(metrics.kv_active_blocks, active_blocks)
                self.assertEqual(metrics.gpu_cache_usage_perc, usage)


if __name__ == "__main__":
    unittest.main()
