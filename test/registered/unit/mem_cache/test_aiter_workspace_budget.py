from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from sglang.srt.layers.attention.aiter_backend import (
    get_aiter_workspace_size_bytes,
)
from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator


class TestAiterWorkspaceBudget(TestCase):
    def _solve(self, *, available, cell_size, context_len, resolver):
        kvc = object.__new__(KVCacheConfigurator)
        kvc.model_config = SimpleNamespace(
            context_len=context_len,
            num_attention_heads=32,
            head_dim=128,
        )

        with (
            patch.object(
                KVCacheConfigurator,
                "config_from_budget",
                lambda self, budget: SimpleNamespace(
                    max_total_num_tokens=budget // cell_size
                ),
            ),
            patch.object(
                KVCacheConfigurator,
                "resolve_max_num_reqs",
                lambda self, tokens, **_kwargs: resolver(tokens),
            ),
            patch(
                "sglang.srt.mem_cache.kv_cache_configurator.get_parallel",
                return_value=SimpleNamespace(attn_tp_size=1),
            ),
        ):
            return kvc._solve_aiter_kv_budget(available)

    def _workspace(self, *, requests, context_len):
        return get_aiter_workspace_size_bytes(
            max_num_reqs=requests,
            num_heads=32,
            max_context_len=context_len,
            head_dim=128,
        )

    def test_workspace_size_matches_backend_allocation_formula(self):
        requests, heads, context_len, head_dim = 17, 32, 262_145, 128
        partitions = (context_len + 255) // 256
        expected = requests * heads * partitions * (head_dim * 4 + 2 * 4)
        self.assertEqual(
            get_aiter_workspace_size_bytes(
                max_num_reqs=requests,
                num_heads=heads,
                max_context_len=context_len,
                head_dim=head_dim,
            ),
            expected,
        )

    def test_reported_shape_fits_workspace_and_kv_in_profiled_budget(self):
        available = 219_721_119_039
        context_len = 262_144
        cell_size = 98_304

        def resolver(tokens):
            return max(min(int(tokens / context_len * 512), 4096), 2048)

        budget = self._solve(
            available=available,
            cell_size=cell_size,
            context_len=context_len,
            resolver=resolver,
        )
        requests = resolver(budget // cell_size)
        workspace = self._workspace(requests=requests, context_len=context_len)
        self.assertLessEqual(budget + workspace, available)
        self.assertGreater(budget, 0)
        self.assertGreater((budget // cell_size), 1_000_000)

    def test_piecewise_request_floor_and_ceiling(self):
        context_len = 1024
        cell_size = 100

        def resolver(tokens):
            return max(min(int(tokens / context_len * 512), 4096), 2048)

        cases = (
            (self._workspace(requests=2048, context_len=context_len) + 300_000, 2048),
            (self._workspace(requests=4096, context_len=context_len) + 1_000_000, 4096),
        )
        for available, expected_requests in cases:
            with self.subTest(expected_requests=expected_requests):
                budget = self._solve(
                    available=available,
                    cell_size=cell_size,
                    context_len=context_len,
                    resolver=resolver,
                )
                requests = resolver(budget // cell_size)
                self.assertEqual(requests, expected_requests)
                self.assertLessEqual(
                    budget
                    + self._workspace(requests=requests, context_len=context_len),
                    available,
                )

    def test_explicit_request_limit_is_still_reserved(self):
        available = 1 << 30
        context_len = 8192
        requested = 123
        budget = self._solve(
            available=available,
            cell_size=4096,
            context_len=context_len,
            resolver=lambda _tokens: requested,
        )
        workspace = self._workspace(requests=requested, context_len=context_len)
        self.assertEqual(budget, available - workspace)


if __name__ == "__main__":
    import unittest

    unittest.main()
