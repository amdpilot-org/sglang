import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.managers.io_struct import SetInternalStateReq
from sglang.srt.managers.schedule_policy import (
    CacheAgnosticPolicy,
    CacheAwarePolicy,
)
from sglang.srt.managers.scheduler import Scheduler


class _TreeCache:
    disable = False

    def supports_fast_match_prefix(self):
        return False


class TestSchedulerRuntimePolicy(unittest.TestCase):
    def setUp(self):
        self.scheduler = object.__new__(Scheduler)
        self.scheduler.schedule_policy = "fcfs"
        self.scheduler.tree_cache = _TreeCache()
        self.scheduler.enable_hierarchical_cache = False
        self.scheduler.enable_priority_scheduling = False
        self.scheduler.schedule_low_priority_values_first = False
        self.scheduler.spec_algorithm = SimpleNamespace(is_none=lambda: True)
        self.scheduler.metrics_reporter = SimpleNamespace(
            spec_total_num_forward_ct=0, spec_total_num_accept_tokens=0
        )
        self.scheduler.waiting_queue = [
            SimpleNamespace(sampling_params=SimpleNamespace(max_new_tokens=2)),
            SimpleNamespace(sampling_params=SimpleNamespace(max_new_tokens=9)),
        ]
        self.scheduler.running_batch = object()
        self.context = Mock()

    def update(self, policy):
        with patch(
            "sglang.srt.managers.scheduler.get_context", return_value=self.context
        ):
            return self.scheduler.set_internal_state(
                SetInternalStateReq(server_args={"schedule_policy": policy})
            )

    def test_accepts_actual_enum_values_and_preserves_live_state(self):
        queue = self.scheduler.waiting_queue
        running_batch = self.scheduler.running_batch
        tree_cache = self.scheduler.tree_cache
        policies = list(CacheAwarePolicy) + list(CacheAgnosticPolicy)

        for policy in policies:
            self.assertTrue(self.update(policy.value).updated)
            self.assertEqual(self.scheduler.schedule_policy, policy.value)

        self.assertIs(self.scheduler.waiting_queue, queue)
        self.assertIs(self.scheduler.running_batch, running_batch)
        self.assertIs(self.scheduler.tree_cache, tree_cache)

    def test_rejects_unknown_value_without_side_effects(self):
        old_policy = self.scheduler.policy = object()

        self.assertFalse(self.update("not-a-policy").updated)

        self.assertEqual(self.scheduler.schedule_policy, "fcfs")
        self.assertIs(self.scheduler.policy, old_policy)
        self.context.override.assert_not_called()

    def test_new_policy_controls_next_queue_sort_in_place(self):
        queue = self.scheduler.waiting_queue

        self.assertTrue(self.update("lof").updated)
        self.scheduler.policy.calc_priority(queue)

        self.assertIs(self.scheduler.waiting_queue, queue)
        self.assertEqual(
            [req.sampling_params.max_new_tokens for req in queue], [9, 2]
        )


if __name__ == "__main__":
    unittest.main()
