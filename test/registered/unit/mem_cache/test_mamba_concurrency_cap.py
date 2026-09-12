"""Regression coverage for the hybrid-state concurrency ceiling.

The scheduler's requested request limit is not necessarily its effective
limit: hybrid models need several Mamba state slots for every running request.
When the state pool is smaller than that requirement, operators must receive a
warning that includes both the effective limit and the slot ratio.
"""

import unittest
from types import SimpleNamespace

from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _resolve(*, requested: int, slots: int, ratio: int) -> int:
    configurator = SimpleNamespace(
        model_config=SimpleNamespace(context_len=4096),
        ps=SimpleNamespace(attn_dp_size=1),
        mambaish_config=object(),
        _calculate_mamba_ratio=lambda: ratio,
    )
    with get_context().override_server_args(
        max_running_requests=requested,
        max_mamba_cache_size=slots,
    ):
        return KVCacheConfigurator.resolve_max_num_reqs(configurator, 100_000)


class TestMambaConcurrencyCap(unittest.TestCase):
    def test_dflash_shaped_shortfall_warns_with_effective_cap(self):
        # Issue #36889's reported shape: 8 requested streams, 10 state slots,
        # and 5 slots consumed per request. The effective concurrency is 2 and
        # the diagnostic must be visible at WARNING, not only at INFO.
        with self.assertLogs(
            "sglang.srt.mem_cache.kv_cache_configurator", level="WARNING"
        ) as logs:
            effective = _resolve(requested=8, slots=10, ratio=5)

        self.assertEqual(effective, 2)
        message = "\n".join(logs.output)
        self.assertIn("max_running_requests is capped to 2", message)
        self.assertIn("max_mamba_cache_size=10", message)
        self.assertIn("5 state slots per request", message)

    def test_exactly_sized_pool_preserves_requested_concurrency(self):
        # Independent boundary: 8 * 5 slots exactly backs all 8 requests, so
        # the Mamba-specific cap warning must not fire.
        with self.assertNoLogs(
            "sglang.srt.mem_cache.kv_cache_configurator", level="WARNING"
        ):
            effective = _resolve(requested=8, slots=40, ratio=5)
        self.assertEqual(effective, 8)

    def test_less_than_one_request_of_state_fails_at_startup(self):
        # Independent lower boundary: silently resolving to zero request slots
        # would produce a server that can admit work but can never execute it.
        with self.assertLogs(
            "sglang.srt.mem_cache.kv_cache_configurator", level="WARNING"
        ):
            with self.assertRaisesRegex(
                RuntimeError, "state cache is too small to serve any requests"
            ):
                _resolve(requested=8, slots=4, ratio=5)


if __name__ == "__main__":
    unittest.main()
