"""Regression tests for anti-starvation aging in the LPM policy."""

import argparse
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.managers.schedule_policy import SchedulePolicy
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _req(rid: str, matched: int):
    return SimpleNamespace(
        rid=rid,
        num_matched_prefix_tokens=matched,
        lpm_waiting_passes=0,
    )


class TestSchedulePolicyLpmAging(CustomTestCase):
    def _policy(self, aging: int) -> SchedulePolicy:
        return SchedulePolicy(
            policy="lpm",
            tree_cache=RadixCache.create_simulated(),
            enable_hierarchical_cache=False,
            enable_priority_scheduling=False,
            schedule_low_priority_values_first=False,
            lpm_aging_tokens_per_pass=aging,
        )

    def test_cold_request_overtakes_continuous_hot_arrivals(self):
        policy = self._policy(25)
        cold = _req("cold", matched=0)

        # Prefix matching is independently exercised elsewhere. Keep the
        # synthetic match lengths fixed here to isolate repeated-pass ordering.
        with patch.object(policy, "_compute_prefix_matches", return_value=set()):
            for pass_index in range(1, 5):
                hot = _req(f"hot-{pass_index}", matched=100)
                queue = [cold, hot]
                policy.calc_priority(queue)
                self.assertIs(queue[0], hot)

            hot = _req("hot-5", matched=100)
            queue = [cold, hot]
            policy.calc_priority(queue)

        self.assertIs(queue[0], cold)

    def test_zero_preserves_lpm_and_does_not_write_counter(self):
        policy = self._policy(0)
        cold = _req("cold", matched=0)
        cold.lpm_waiting_passes = 10_000
        hot = _req("hot", matched=100)

        with patch.object(policy, "_compute_prefix_matches", return_value=set()):
            policy.calc_priority([cold, hot])

        self.assertEqual(cold.lpm_waiting_passes, 10_000)
        queue = [cold, hot]
        SchedulePolicy._sort_by_longest_prefix(queue, set())
        self.assertEqual([req.rid for req in queue], ["hot", "cold"])

    def test_exact_threshold_is_stable_and_next_pass_overtakes(self):
        cold = _req("cold", matched=0)
        hot = _req("hot", matched=100)
        cold.lpm_waiting_passes = 4

        queue = [hot, cold]
        SchedulePolicy._sort_by_longest_prefix(queue, set(), 25)
        self.assertEqual([req.rid for req in queue], ["hot", "cold"])

        cold.lpm_waiting_passes += 1
        SchedulePolicy._sort_by_longest_prefix(queue, set(), 25)
        self.assertEqual([req.rid for req in queue], ["cold", "hot"])

    def test_temporary_deprioritization_still_wins_over_age(self):
        cold = _req("cold", matched=0)
        cold.lpm_waiting_passes = 1_000_000
        hot = _req("hot", matched=1)
        queue = [cold, hot]

        SchedulePolicy._sort_by_longest_prefix(queue, {"cold"}, 25)

        self.assertEqual([req.rid for req in queue], ["hot", "cold"])

    def test_large_queue_fcfs_fallback_does_not_accumulate_lpm_age(self):
        policy = self._policy(25)
        queue = [_req(str(index), matched=index) for index in range(129)]

        policy.calc_priority(queue)

        self.assertTrue(all(req.lpm_waiting_passes == 0 for req in queue))


class TestLpmAgingServerArg(CustomTestCase):
    def test_default_is_disabled(self):
        from sglang.srt.arg_groups.fields.schedule import Schedule

        self.assertEqual(Schedule().lpm_aging_tokens_per_pass, 0)

    def test_cli_value_is_wired_to_server_args(self):
        from sglang.srt.server_args import ServerArgs

        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        parsed = parser.parse_args(
            ["--model", "dummy", "--lpm-aging-tokens-per-pass", "4096"]
        )

        server_args = ServerArgs.from_cli_args(parsed)
        self.assertEqual(server_args.lpm_aging_tokens_per_pass, 4096)


if __name__ == "__main__":
    unittest.main()
