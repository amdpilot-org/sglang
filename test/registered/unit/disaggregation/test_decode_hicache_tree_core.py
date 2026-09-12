"""Unit tests for decode HiCache TreeCore interactions."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import torch

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.decode_hicache_mixin import (
    DecodeHiCachePreallocMixin,
    DecodeHiCacheTransferMixin,
    DecodePrefixMatch,
    HiCacheRestoreGatedKVReceiver,
    HiCacheRestoreResult,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestDecodeHiCacheTreeCore(CustomTestCase):
    def test_storage_probe_and_prefetch_use_node_handles(self):
        ongoing_prefetch = {}

        def register_prefetch(req_id, *_args, **_kwargs):
            ongoing_prefetch[req_id] = object()

        tree_cache = SimpleNamespace(
            hicache_storage_pass_prefix_keys=True,
            ongoing_prefetch=ongoing_prefetch,
            is_backuped=Mock(return_value=True),
            is_root=Mock(return_value=False),
            get_last_hash_value=Mock(return_value="h2"),
            get_prefix_hash_values=Mock(return_value=["h0", "h1"]),
            query_storage_hit_length=Mock(return_value=2),
            prefetch_from_storage=Mock(side_effect=register_prefetch),
        )
        harness = SimpleNamespace(
            scheduler=SimpleNamespace(enable_decode_hicache=True),
            tree_cache=tree_cache,
        )
        req = SimpleNamespace(
            rid="req-0",
            origin_input_ids=[0, 1, 2, 3, 4, 5, 6, 7],
            extra_key="model",
            cache_salt="tenant-a",
        )
        result = SimpleNamespace(
            device_indices=torch.tensor([10, 11]),
            host_hit_length=2,
            last_device_node=11,
            last_host_node=22,
        )

        prefix_match = DecodeHiCachePreallocMixin._build_decode_prefix_match(
            harness, req, result
        )

        self.assertEqual(prefix_match.l3_storage_hit_length, 2)
        tree_cache.query_storage_hit_length.assert_called_once_with(
            22,
            [4, 5, 6, 7],
            "h2",
            ["h0", "h1"],
            extra_key="model",
            cache_salt="tenant-a",
        )

        DecodeHiCachePreallocMixin._start_hicache_prefetch(harness, req, prefix_match)

        self.assertTrue(prefix_match.prefetch_registered)
        tree_cache.prefetch_from_storage.assert_called_once_with(
            "req-0",
            22,
            [4, 5],
            "h2",
            ["h0", "h1"],
            extra_key="model",
            cache_salt="tenant-a",
        )

    def test_stale_prefetch_anchor_degrades_to_l2(self):
        tree_cache = SimpleNamespace(
            hicache_storage_pass_prefix_keys=True,
            ongoing_prefetch={},
            get_last_hash_value=Mock(side_effect=KeyError(22)),
            get_prefix_hash_values=Mock(),
            prefetch_from_storage=Mock(),
        )
        harness = SimpleNamespace(tree_cache=tree_cache)
        req = SimpleNamespace(
            rid="req-0",
            origin_input_ids=[0, 1, 2, 3, 4, 5],
            extra_key=None,
            cache_salt=None,
        )
        prefix_match = DecodePrefixMatch(
            prefix_indices=torch.tensor([10, 11]),
            l2_host_hit_length=2,
            l3_storage_hit_length=2,
            last_device_node=11,
            last_host_node=22,
        )

        DecodeHiCachePreallocMixin._start_hicache_prefetch(harness, req, prefix_match)

        self.assertEqual(prefix_match.l3_storage_hit_length, 0)
        self.assertFalse(prefix_match.prefetch_registered)
        tree_cache.get_prefix_hash_values.assert_not_called()
        tree_cache.prefetch_from_storage.assert_not_called()

    def test_l3_only_restore_waits_for_prefetch_before_load_back(self):
        """A decode restart can have an L3 hit with no L1/L2 coverage."""
        tree_cache = MagicMock()
        tree_cache.check_prefetch_progress.return_value = False
        harness = SimpleNamespace(tree_cache=tree_cache)
        decode_req = SimpleNamespace(
            req=SimpleNamespace(rid="req-restart", origin_input_ids=list(range(64))),
            prefix_match=DecodePrefixMatch(
                prefix_indices=torch.empty(0, dtype=torch.int64),
                l2_host_hit_length=0,
                l3_storage_hit_length=64,
                last_device_node=tree_cache.root_node,
            ),
            hicache_restore_status=HiCacheRestoreResult.PENDING,
        )

        queued = DecodeHiCacheTransferMixin._try_hicache_queue_load_back(
            harness, decode_req
        )

        self.assertFalse(queued)
        self.assertEqual(decode_req.hicache_restore_status, HiCacheRestoreResult.PENDING)
        tree_cache.pop_prefetch_loaded_tokens.assert_not_called()
        tree_cache.init_load_back.assert_not_called()

    @patch("sglang.srt.disaggregation.decode_hicache_mixin.match_prefix_for_req")
    def test_l3_only_restore_queues_complete_coverage(self, match_prefix):
        tree_cache = MagicMock()
        tree_cache.check_prefetch_progress.return_value = True
        restored_indices = torch.arange(100, 164, dtype=torch.int64)
        restored_node = object()
        lock_receipt = object()
        tree_cache.init_load_back.return_value = (restored_indices, restored_node)
        tree_cache.inc_lock_ref.return_value.to_dec_params.return_value = lock_receipt
        match_prefix.return_value = SimpleNamespace(
            device_indices=torch.empty(0, dtype=torch.int64),
            best_match_node=tree_cache.root_node,
            host_hit_length=64,
        )
        req = SimpleNamespace(
            rid="req-restart",
            origin_input_ids=list(range(64)),
            last_node=tree_cache.root_node,
        )
        decode_req = SimpleNamespace(
            req=req,
            prefix_match=DecodePrefixMatch(
                prefix_indices=torch.empty(0, dtype=torch.int64),
                l2_host_hit_length=0,
                l3_storage_hit_length=64,
                last_device_node=tree_cache.root_node,
            ),
            hicache_restore_status=HiCacheRestoreResult.PENDING,
        )
        harness = SimpleNamespace(tree_cache=tree_cache)

        queued = DecodeHiCacheTransferMixin._try_hicache_queue_load_back(
            harness, decode_req
        )

        self.assertTrue(queued)
        self.assertEqual(decode_req.hicache_restore_status, HiCacheRestoreResult.PENDING)
        torch.testing.assert_close(
            decode_req.hicache_restored_kv_indices, restored_indices
        )
        self.assertIs(decode_req.hicache_restored_node, restored_node)
        self.assertIs(decode_req.hicache_restore_lock_receipt, lock_receipt)
        tree_cache.pop_prefetch_loaded_tokens.assert_called_once_with("req-restart")

    def test_transfer_success_is_gated_until_l3_restore_is_ready(self):
        receiver = MagicMock()
        receiver.poll.return_value = KVPoll.Success
        decode_req = SimpleNamespace(
            kv_receiver=receiver,
            hicache_restore_status=HiCacheRestoreResult.PENDING,
        )
        gated = HiCacheRestoreGatedKVReceiver(decode_req)

        self.assertEqual(gated.poll(), KVPoll.Transferring)
        decode_req.hicache_restore_status = HiCacheRestoreResult.READY
        self.assertEqual(gated.poll(), KVPoll.Success)


if __name__ == "__main__":
    unittest.main()
