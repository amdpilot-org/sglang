"""GPU regression for accepted-token KV cache relocation.

The case isolates movement after acceptance decisions: it does not construct
acceptance results or tree masks. It covers zero, partial, and full draft
acceptance, short -1 tails, boundary accept indices, and disjoint source and
destination slot maps.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.memory_pool import move_kv_cache_native
from sglang.srt.speculative.spec_utils import move_accept_tokens_to_target_kvcache
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


class _RecordingKVCache:
    def __init__(self, k_cache, v_cache):
        self.k_cache = k_cache
        self.v_cache = v_cache
        self.tgt_loc = None
        self.src_loc = None

    def move_kv_cache(self, tgt_loc, src_loc):
        self.tgt_loc = tgt_loc.clone()
        self.src_loc = src_loc.clone()
        move_kv_cache_native([self.k_cache], [self.v_cache], tgt_loc, src_loc)


class _RecordingAllocator:
    def __init__(self, kv_cache):
        self.kv_cache = kv_cache

    def get_kvcache(self):
        return self.kv_cache


class TestMoveAcceptTokensToTargetKVCache(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA or ROCm")
    def test_moves_exact_accepted_cache_content(self):
        device = torch.device("cuda")
        req_to_token = torch.full((3, 16), -1, dtype=torch.int64, device=device)
        req_to_token[0, 4] = 8
        req_to_token[1, 7:9] = torch.tensor([9, 10], device=device)
        req_to_token[2, 9:14] = torch.arange(11, 16, device=device)

        batch = SimpleNamespace(
            seq_lens=torch.tensor([4, 7, 9], dtype=torch.int64, device=device),
            req_pool_indices=torch.tensor([0, 1, 2], dtype=torch.int64, device=device),
            req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
            out_cache_loc=torch.arange(20, 35, dtype=torch.int64, device=device),
        )
        accept_index = torch.tensor(
            [
                [0, -1, -1, -1, -1],
                [1, 2, -1, -1, -1],
                [0, 7, 14, 3, 4],
            ],
            dtype=torch.int64,
            device=device,
        )
        num_correct_drafts = torch.tensor([0, 1, 4], dtype=torch.int64, device=device)

        k_cache = torch.arange(35 * 6, dtype=torch.float32, device=device).reshape(
            35, 2, 3
        )
        v_cache = -torch.arange(35 * 6, dtype=torch.float32, device=device).reshape(
            35, 2, 3
        )
        expected_k = k_cache.cpu().clone()
        expected_v = v_cache.cpu().clone()
        expected_tgt = torch.arange(8, 16, dtype=torch.int64)
        expected_src = torch.tensor([20, 21, 22, 20, 27, 34, 23, 24])
        expected_k[expected_tgt] = expected_k[expected_src]
        expected_v[expected_tgt] = expected_v[expected_src]

        kv_cache = _RecordingKVCache(k_cache, v_cache)
        allocator = _RecordingAllocator(kv_cache)
        move_accept_tokens_to_target_kvcache(
            batch, accept_index, num_correct_drafts, allocator
        )
        torch.cuda.synchronize()

        valid_accept_index = accept_index != -1
        self.assertEqual(int(accept_index.min().item()), -1)
        self.assertEqual(
            int(accept_index.max().item()), int(batch.out_cache_loc.numel() - 1)
        )
        self.assertTrue(bool((accept_index[valid_accept_index] >= 0).all().item()))
        self.assertFalse(set(expected_tgt.tolist()) & set(expected_src.tolist()))
        expected_tgt_device = expected_tgt.to(device)
        expected_src_device = expected_src.to(device)
        self.assertTrue(
            torch.equal(kv_cache.tgt_loc[: expected_tgt.numel()], expected_tgt_device)
        )
        self.assertTrue(
            torch.equal(kv_cache.src_loc[: expected_src.numel()], expected_src_device)
        )
        torch.testing.assert_close(k_cache.cpu(), expected_k, rtol=0, atol=0)
        torch.testing.assert_close(v_cache.cpu(), expected_v, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
