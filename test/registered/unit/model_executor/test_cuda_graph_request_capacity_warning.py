import unittest
from types import MethodType, SimpleNamespace
from unittest.mock import patch

from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestCudaGraphRequestCapacityWarning(unittest.TestCase):
    def _resolve(
        self,
        *,
        max_running_requests=None,
        graph_backend="full",
        graph_max_bs=32,
        repetitions=1,
    ):
        configurator = SimpleNamespace(
            model_config=SimpleNamespace(context_len=32768),
            ps=SimpleNamespace(attn_dp_size=1),
            mambaish_config=None,
            _warned_cuda_graph_request_capacity=False,
        )
        configurator._warn_cuda_graph_request_capacity = MethodType(
            KVCacheConfigurator._warn_cuda_graph_request_capacity, configurator
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
            patch(
                "sglang.srt.mem_cache.kv_cache_configurator.logger.warning"
            ) as warning,
        ):
            for _ in range(repetitions):
                resolved = KVCacheConfigurator.resolve_max_num_reqs(
                    configurator, token_capacity=3_145_118
                )
        return resolved, warning

    def test_warns_for_issue_default_gap(self):
        resolved, warning = self._resolve(repetitions=2)

        self.assertEqual(resolved, 4096)
        warning.assert_called_once()
        message, max_requests, graph_max_bs = warning.call_args.args
        self.assertIn("exceeds the largest captured decode CUDA graph", message)
        self.assertEqual((max_requests, graph_max_bs), (4096, 32))

    def test_equal_capacity_does_not_warn(self):
        resolved, warning = self._resolve(max_running_requests=32)

        self.assertEqual(resolved, 32)
        warning.assert_not_called()

    def test_disabled_graph_does_not_warn(self):
        resolved, warning = self._resolve(graph_backend="disabled")

        self.assertEqual(resolved, 4096)
        warning.assert_not_called()


if __name__ == "__main__":
    unittest.main()
