# SPDX-License-Identifier: Apache-2.0
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.multimodal_gen.configs.sample.teacache import TeaCacheParams
from sglang.multimodal_gen.runtime.cache.teacache import TeaCacheMixin


class _DummyTeaCache(TeaCacheMixin):
    prefix = "wan"


class TestTeaCacheCFGLifecycle(unittest.TestCase):
    def setUp(self) -> None:
        self.cache = _DummyTeaCache()
        self.cache._init_teacache_state()
        self.batch = SimpleNamespace(
            enable_teacache=True,
            teacache_params=TeaCacheParams(teacache_thresh=0.1, coefficients=[1.0]),
            num_inference_steps=50,
            do_classifier_free_guidance=True,
            is_cfg_negative=False,
        )
        self.forward_context = SimpleNamespace(
            current_timestep=0, forward_batch=self.batch
        )
        self.context_patch = patch(
            "sglang.multimodal_gen.runtime.managers.forward_context.get_forward_context",
            return_value=self.forward_context,
        )
        self.context_patch.start()
        self.addCleanup(self.context_patch.stop)

    def _context(self, *, cfg_parallel: bool):
        with patch(
            "sglang.multimodal_gen.runtime.server_args.get_global_server_args",
            return_value=SimpleNamespace(enable_cfg_parallel=cfg_parallel),
        ):
            return self.cache._get_teacache_context()

    def test_serial_cfg_negative_branch_does_not_reset_twice(self) -> None:
        self._context(cfg_parallel=False)
        self.cache.cnt = 1

        self.batch.is_cfg_negative = True
        ctx = self._context(cfg_parallel=False)

        self.assertTrue(ctx.is_cfg_negative)
        self.assertEqual(self.cache.cnt, 1)

    def test_cfg_parallel_negative_rank_resets_each_request(self) -> None:
        self.cache.cnt = 50
        self.cache.is_cfg_negative = True
        self.cache.previous_modulated_input_negative = torch.tensor([123.0])
        self.cache.previous_residual_negative = torch.tensor([456.0])
        self.cache.accumulated_rel_l1_distance_negative = 7.0
        self.batch.is_cfg_negative = True

        ctx = self._context(cfg_parallel=True)

        self.assertTrue(ctx.cfg_parallel)
        self.assertEqual(self.cache.cnt, 0)
        self.assertIsNone(self.cache.previous_modulated_input_negative)
        self.assertIsNone(self.cache.previous_residual_negative)
        self.assertEqual(self.cache.accumulated_rel_l1_distance_negative, 0.0)

    def test_serial_cfg_next_request_resets_after_negative_branch(self) -> None:
        self.cache.cnt = 50
        self.cache.is_cfg_negative = True
        self.cache.previous_modulated_input = torch.tensor([123.0])
        self.batch.is_cfg_negative = False

        self._context(cfg_parallel=False)

        self.assertEqual(self.cache.cnt, 0)
        self.assertIsNone(self.cache.previous_modulated_input)

    def test_noninitial_timestep_never_resets(self) -> None:
        self.forward_context.current_timestep = 1
        self.cache.cnt = 3
        self.batch.is_cfg_negative = True

        self._context(cfg_parallel=True)

        self.assertEqual(self.cache.cnt, 3)

    def test_skip_boundaries_follow_local_cfg_topology(self) -> None:
        params = TeaCacheParams(start_skipping=5, end_skipping=-1)

        self.assertEqual(params.get_skip_boundaries(50, do_cfg=False), (5, 49))
        self.assertEqual(params.get_skip_boundaries(50, do_cfg=True), (10, 98))
        self.assertEqual(
            params.get_skip_boundaries(50, do_cfg=True, cfg_parallel=True),
            (5, 49),
        )


if __name__ == "__main__":
    unittest.main()
