"""Regression tests for HiCache prefetch timeout termination."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestHiRadixPrefetchTimeout(unittest.TestCase):
    def _make_cache(self, policy: str, *, timed_out: bool):
        cache = object.__new__(HiRadixCache)
        cache.prefetch_stop_policy = policy
        cache.is_prefetch_timeout = MagicMock(return_value=timed_out)
        return cache

    @staticmethod
    def _make_operation(*, pool_transfers_done: bool):
        return SimpleNamespace(
            hash_value=["page"],
            completed_tokens=1,
            pool_transfers=[object()],
            pool_transfers_done=pool_transfers_done,
        )

    def test_timeout_terminates_with_pending_pool_transfers(self):
        cache = self._make_cache("timeout", timed_out=True)
        operation = self._make_operation(pool_transfers_done=False)

        self.assertTrue(cache.can_terminate_prefetch(operation))
        cache.is_prefetch_timeout.assert_called_once_with(operation)

    def test_timeout_terminates_with_completed_pool_transfers(self):
        cache = self._make_cache("timeout", timed_out=True)
        operation = self._make_operation(pool_transfers_done=True)

        self.assertTrue(cache.can_terminate_prefetch(operation))

    def test_before_timeout_does_not_terminate(self):
        cache = self._make_cache("timeout", timed_out=False)
        operation = self._make_operation(pool_transfers_done=False)

        self.assertFalse(cache.can_terminate_prefetch(operation))

    def test_non_timeout_policy_boundaries(self):
        operation = self._make_operation(pool_transfers_done=False)

        self.assertTrue(
            self._make_cache("best_effort", timed_out=False).can_terminate_prefetch(
                operation
            )
        )
        self.assertFalse(
            self._make_cache("wait_complete", timed_out=True).can_terminate_prefetch(
                operation
            )
        )


if __name__ == "__main__":
    unittest.main()
