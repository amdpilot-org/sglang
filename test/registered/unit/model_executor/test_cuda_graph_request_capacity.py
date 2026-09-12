import unittest
from types import MethodType, SimpleNamespace
from unittest.mock import patch

from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestCudaGraphRequestCapacity(unittest.TestCase):
    def _resolve(
        self,
        *,
        max_running_requests=None,
        graph_backend="full",
        graph_max_bs=32,
        attn_dp_size=1,
        token_capacity=3_145_118,
    ):
        configurator = SimpleNamespace(
            model_config=SimpleNamespace(context_len=32768),
            ps=SimpleNamespace(attn_dp_size=attn_dp_size),
            mambaish_config=None,
            _warned_cuda_graph_request_capacity=False,
            # This makes the policy regression runnable on the recorded base,
            # where the candidate's new private warning helper does not exist.
            _warn_cuda_graph_request_capacity=lambda _max_num_reqs: None,
        )
        schedule = SimpleNamespace(max_running_requests=max_running_requests)
        execution = SimpleNamespace(
            graph=SimpleNamespace(
                cuda_graph_config=SimpleNamespace(
                    decode=SimpleNamespace(
                        backend=graph_backend,
                        max_bs=graph_max_bs,
                    )
                )
            )
        )
        with (
            patch(
                "sglang.srt.mem_cache.kv_cache_configurator.get_schedule",
                return_value=schedule,
            ),
            patch(
                "sglang.srt.mem_cache.kv_cache_configurator.get_exec",
                return_value=execution,
            ),
        ):
            return KVCacheConfigurator.resolve_max_num_reqs(
                configurator, token_capacity=token_capacity
            )

    def test_issue_defaults_cap_admission_at_graph_coverage(self):
        self.assertEqual(self._resolve(), 32)

    def test_explicit_admission_override_is_preserved(self):
        self.assertEqual(self._resolve(max_running_requests=4096), 4096)

    def test_disabled_graph_keeps_capacity_derived_default(self):
        self.assertEqual(self._resolve(graph_backend="disabled"), 4096)

    def test_missing_graph_limit_keeps_capacity_derived_default(self):
        self.assertEqual(self._resolve(graph_max_bs=None), 4096)

    def test_dp_worker_uses_local_explicit_capacity(self):
        self.assertEqual(self._resolve(max_running_requests=128, attn_dp_size=4), 32)

    def test_explicit_overflow_retains_candidate_warning_once(self):
        configurator = SimpleNamespace(
            _warned_cuda_graph_request_capacity=False,
        )
        configurator._warn_cuda_graph_request_capacity = MethodType(
            KVCacheConfigurator._warn_cuda_graph_request_capacity, configurator
        )
        execution = SimpleNamespace(
            graph=SimpleNamespace(
                cuda_graph_config=SimpleNamespace(
                    decode=SimpleNamespace(backend="full", max_bs=32)
                )
            )
        )
        with (
            patch(
                "sglang.srt.mem_cache.kv_cache_configurator.get_exec",
                return_value=execution,
            ),
            patch(
                "sglang.srt.mem_cache.kv_cache_configurator.logger.warning"
            ) as warning,
        ):
            configurator._warn_cuda_graph_request_capacity(4096)
            configurator._warn_cuda_graph_request_capacity(4096)

        warning.assert_called_once()
        self.assertEqual(warning.call_args.args[1:], (4096, 32))


if __name__ == "__main__":
    unittest.main()
