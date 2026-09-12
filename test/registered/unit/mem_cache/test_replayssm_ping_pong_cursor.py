"""Regression tests for ReplaySSM cursors on extra-buffer slot ownership."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.arg_groups.attention_hook import handle_linear_attn_backend
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
from sglang.srt.runtime_context import override_platform
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Allocator:
    def __init__(self, slots):
        self.slots = torch.tensor(slots, dtype=torch.int64)

    def alloc(self, size):
        assert size == len(self.slots)
        return self.slots.clone()


def _request(req_pool_idx=1):
    return SimpleNamespace(
        rid="test",
        kv=SimpleNamespace(
            req_pool_idx=req_pool_idx,
            mamba_ping_pong_track_buffer=None,
            mamba_next_track_idx=None,
            mamba_last_track_idx=None,
        ),
    )


class TestReplaySSMPingPongCursor(unittest.TestCase):
    def _pool(self, slots=(3, 4)):
        pool = object.__new__(HybridReqToTokenPool)
        pool.mamba_ping_pong_track_buffer_size = len(slots)
        pool.enable_mamba_extra_buffer_lazy = False
        pool.mamba_allocator = _Allocator(slots)
        pool.mamba_pool = SimpleNamespace(
            replayssm_write_pos=torch.tensor([0, 0, 0, 5, 7, 9], dtype=torch.int32)
        )
        pool.req_index_to_mamba_ping_pong_track_buffer_mapping = torch.zeros(
            (4, len(slots)), dtype=torch.int64
        )
        return pool

    def test_allocated_tracking_slots_clear_recycled_cursors(self):
        pool = self._pool()
        req = _request()

        pool._alloc_ping_pong_buffer(req)

        self.assertEqual(req.kv.mamba_ping_pong_track_buffer.tolist(), [3, 4])
        self.assertEqual(pool.mamba_pool.replayssm_write_pos[[3, 4]].tolist(), [0, 0])

    def test_donation_replacement_clears_recycled_cursor(self):
        pool = self._pool(slots=(3, 4))
        req = _request()
        req.kv.mamba_ping_pong_track_buffer = torch.tensor([3, 4], dtype=torch.int64)
        req.kv.mamba_next_track_idx = 1
        req.kv.mamba_last_track_idx = 0

        donated = pool.donate_mamba_ping_pong_slot(req, torch.tensor([5]))

        self.assertEqual(donated.item(), 3)
        self.assertEqual(req.kv.mamba_ping_pong_track_buffer.tolist(), [5, 4])
        self.assertEqual(pool.mamba_pool.replayssm_write_pos[5].item(), 0)

    def test_non_replayssm_pool_leaves_cursor_bookkeeping_absent(self):
        pool = self._pool()
        pool.mamba_pool.replayssm_write_pos = None
        req = _request()

        pool._alloc_ping_pong_buffer(req)
        pool.set_mamba_ping_pong_slot(req, 0, -1)

        self.assertEqual(req.kv.mamba_ping_pong_track_buffer.tolist(), [-1, 4])


class TestReplaySSMExtraBufferGuard(unittest.TestCase):
    def _args(self):
        return ServerArgs(
            model_path="dummy",
            enable_linear_replayssm=True,
            mamba_radix_cache_strategy="extra_buffer",
        )

    def test_gdn_extra_buffer_is_admitted(self):
        with (
            override_platform(is_cuda=False),
            patch(
                "sglang.srt.arg_groups.attention_hook._replayssm_is_kda",
                return_value=False,
            ),
            patch(
                "sglang.srt.arg_groups.attention_hook.model_config_of",
                return_value=SimpleNamespace(),
            ),
        ):
            handle_linear_attn_backend(self._args())

    def test_kda_extra_buffer_remains_rejected(self):
        with (
            override_platform(is_cuda=False),
            patch(
                "sglang.srt.arg_groups.attention_hook._replayssm_is_kda",
                return_value=True,
            ),
            patch(
                "sglang.srt.arg_groups.attention_hook.model_config_of",
                return_value=SimpleNamespace(),
            ),
            self.assertRaisesRegex(ValueError, "KDA model requires"),
        ):
            handle_linear_attn_backend(self._args())


if __name__ == "__main__":
    unittest.main()
