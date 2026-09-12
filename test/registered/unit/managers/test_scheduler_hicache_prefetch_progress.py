import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler import Scheduler

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestSchedulerHiCachePrefetchProgress(unittest.TestCase):
    @staticmethod
    def _scheduler(*, enabled=True, results=None):
        scheduler = object.__new__(Scheduler)
        scheduler.enable_hicache_storage = enabled
        scheduler.waiting_queue = [
            SimpleNamespace(rid="request-0"),
            SimpleNamespace(rid="request-1"),
            SimpleNamespace(rid="request-2"),
        ]
        results = results or {}
        scheduler.tree_cache = SimpleNamespace(
            check_prefetch_progress=Mock(
                side_effect=lambda rid: results.get(rid, True)
            )
        )
        return scheduler

    def test_checks_entire_waiting_queue_before_rank_local_admission(self):
        # Model two TP ranks whose local capacity would let their admission loops
        # inspect different numbers of requests.  Progress synchronization is no
        # longer gated by those limits: both ranks check the full broadcast queue.
        checked_by_rank = []
        for local_capacity in (1, 2):
            scheduler = self._scheduler()
            progress = scheduler._check_waiting_queue_prefetch_progress()
            admitted = [
                req.rid
                for req in scheduler.waiting_queue[:local_capacity]
                if progress[req.rid]
            ]
            self.assertEqual(len(admitted), local_capacity)
            checked_by_rank.append(
                [
                    call.args[0]
                    for call in scheduler.tree_cache.check_prefetch_progress.call_args_list
                ]
            )

        self.assertEqual(
            checked_by_rank,
            [
                ["request-0", "request-1", "request-2"],
                ["request-0", "request-1", "request-2"],
            ],
        )

    def test_snapshots_before_batch_full_early_return(self):
        scheduler = self._scheduler()
        scheduler.grammar_manager = SimpleNamespace(
            has_waiting_grammars=lambda: False
        )
        scheduler.enable_hierarchical_cache = False
        scheduler.enable_unified_cache_external_linker = False
        scheduler.enable_priority_preemption = False
        scheduler.is_hybrid_swa = False
        scheduler.chunked_req = None
        scheduler._check_waiting_queue_prefetch_progress = Mock(
            return_value={req.rid: True for req in scheduler.waiting_queue}
        )
        running_batch = SimpleNamespace(batch_is_full=True)

        with patch(
            "sglang.srt.managers.scheduler.get_memory",
            return_value=SimpleNamespace(enable_flexkv=False),
        ):
            new_batch, returned_running_batch = scheduler._get_new_batch_prefill_raw(
                None, running_batch
            )

        self.assertIsNone(new_batch)
        self.assertIs(returned_running_batch, running_batch)
        scheduler._check_waiting_queue_prefetch_progress.assert_called_once_with()

    def test_preserves_unfinished_results_for_local_admission(self):
        scheduler = self._scheduler(results={"request-1": False})

        self.assertEqual(
            scheduler._check_waiting_queue_prefetch_progress(),
            {"request-0": True, "request-1": False, "request-2": True},
        )

    def test_disabled_storage_does_not_check_progress(self):
        scheduler = self._scheduler(enabled=False)

        self.assertIsNone(scheduler._check_waiting_queue_prefetch_progress())
        scheduler.tree_cache.check_prefetch_progress.assert_not_called()

    def test_empty_waiting_queue_returns_empty_snapshot(self):
        scheduler = self._scheduler()
        scheduler.waiting_queue = []

        self.assertEqual(scheduler._check_waiting_queue_prefetch_progress(), {})
        scheduler.tree_cache.check_prefetch_progress.assert_not_called()


if __name__ == "__main__":
    unittest.main()
