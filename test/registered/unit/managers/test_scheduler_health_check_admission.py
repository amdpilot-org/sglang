from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler import Scheduler

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestSchedulerHealthCheckAdmission(CustomTestCase):
    def _scheduler(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.scheduler_stage_metrics = None
        scheduler.session_controller = SimpleNamespace(maybe_reap=MagicMock())
        scheduler.return_health_check_ipcs = deque()
        scheduler.is_fully_idle = MagicMock(return_value=True)
        scheduler.ipc_channels = SimpleNamespace(
            send_to_tokenizer=SimpleNamespace(send_output=MagicMock()),
            recv_from_rpc=None,
        )
        scheduler.rust_server = None
        scheduler.flush_wrapper = SimpleNamespace(check_pending=MagicMock())
        scheduler.external_corpus_manager = None
        scheduler.waiting_queue = []
        scheduler.attempted_prefill_batch_sizes = []
        scheduler.prefill_delayer_outcomes = {"mixed/delay": 0, "all/no_wait": 0}
        scheduler.prefill_progress = 0

        def ordinary_admission(req):
            scheduler.waiting_queue.append(req)
            scheduler.attempted_prefill_batch_sizes.append(len(scheduler.waiting_queue))
            scheduler.prefill_delayer_outcomes["mixed/delay"] += 1
            scheduler.prefill_progress += len(req.input_ids)

        scheduler._request_dispatcher = MagicMock(side_effect=ordinary_admission)
        return scheduler

    @patch("sglang.srt.managers.scheduler.get_mm")
    def test_idle_health_probe_enters_generation_admission(self, get_mm):
        get_mm.return_value.mm_feature_transport = None
        scheduler = self._scheduler()
        health = SimpleNamespace(
            rid="HEALTH_CHECK_idle",
            input_ids=[0],
            http_worker_ipc="worker-ipc",
        )

        scheduler.process_input_requests([health])

        scheduler._request_dispatcher.assert_called_once_with(health)
        self.assertEqual(scheduler.waiting_queue, [health])
        self.assertEqual(scheduler.attempted_prefill_batch_sizes, [1])
        self.assertEqual(
            scheduler.prefill_delayer_outcomes,
            {"mixed/delay": 1, "all/no_wait": 0},
        )
        self.assertEqual(scheduler.prefill_progress, 1)
        self.assertEqual(list(scheduler.return_health_check_ipcs), [])
        scheduler.ipc_channels.send_to_tokenizer.send_output.assert_not_called()

    @patch("sglang.srt.managers.scheduler.get_mm")
    def test_idle_probe_and_user_are_both_admitted(self, get_mm):
        get_mm.return_value.mm_feature_transport = None
        scheduler = self._scheduler()
        health = SimpleNamespace(
            rid="HEALTH_CHECK_interleaved",
            input_ids=[0],
            http_worker_ipc="worker-ipc",
        )
        user = SimpleNamespace(rid="user", input_ids=[1, 2], http_worker_ipc=None)

        scheduler.process_input_requests([health, user])

        self.assertEqual(
            scheduler._request_dispatcher.call_args_list,
            [call(health), call(user)],
        )
        self.assertEqual(scheduler.waiting_queue, [health, user])
        self.assertEqual(scheduler.attempted_prefill_batch_sizes, [1, 2])
        self.assertEqual(
            scheduler.prefill_delayer_outcomes,
            {"mixed/delay": 2, "all/no_wait": 0},
        )
        self.assertEqual(scheduler.prefill_progress, 3)

    @patch("sglang.srt.managers.scheduler.get_mm")
    def test_busy_probe_keeps_deferred_health_signal(self, get_mm):
        get_mm.return_value.mm_feature_transport = None
        scheduler = self._scheduler()
        scheduler.is_fully_idle.return_value = False
        health = SimpleNamespace(
            rid="HEALTH_CHECK_busy",
            input_ids=[0],
            http_worker_ipc="worker-ipc",
        )

        scheduler.process_input_requests([health])

        scheduler._request_dispatcher.assert_not_called()
        self.assertEqual(list(scheduler.return_health_check_ipcs), ["worker-ipc"])
        scheduler.ipc_channels.send_to_tokenizer.send_output.assert_not_called()


if __name__ == "__main__":
    import unittest

    unittest.main()
