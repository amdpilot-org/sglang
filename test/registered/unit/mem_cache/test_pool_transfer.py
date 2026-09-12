import dataclasses
import json
import pickle
import unittest

from sglang.srt.mem_cache import hicache_storage, pool_transfer
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestPoolTransferCompatibility(unittest.TestCase):
    def test_legacy_exports_preserve_type_identity(self):
        for name in (
            "PoolName",
            "PoolHitPolicy",
            "PoolTransfer",
            "PoolTransferResult",
        ):
            self.assertIs(getattr(hicache_storage, name), getattr(pool_transfer, name))

    def test_serialization_behavior(self):
        transfer = pool_transfer.PoolTransfer(
            name=pool_transfer.PoolName.KV,
            keys=["page-0"],
            hit_policy=pool_transfer.PoolHitPolicy.ALL_PAGES,
        )
        result = pool_transfer.PoolTransferResult(
            1, {pool_transfer.PoolName.MAMBA: 2}, [1]
        )

        self.assertEqual(pickle.loads(pickle.dumps(transfer)), transfer)
        self.assertEqual(pickle.loads(pickle.dumps(result)), result)
        legacy_pickle = pickle.dumps(transfer, protocol=0).replace(
            b"sglang.srt.mem_cache.pool_transfer\nPoolTransfer\n",
            b"sglang.srt.mem_cache.hicache_storage\nPoolTransfer\n",
        )
        self.assertEqual(pickle.loads(legacy_pickle), transfer)
        self.assertEqual(
            json.dumps(
                {
                    "name": pool_transfer.PoolName.KV,
                    "policy": pool_transfer.PoolHitPolicy.ALL_PAGES,
                }
            ),
            '{"name": "kv", "policy": "all_pages"}',
        )
        self.assertEqual(dataclasses.asdict(transfer)["keys"], ["page-0"])

    def test_transfer_result_behavior(self):
        result = pool_transfer.PoolTransferResult.empty()
        result.update_kv_hit_pages(3)
        result.update_kv_hit_pages(2)
        result.update_extra_pool_hit_pages({pool_transfer.PoolName.MAMBA: 1})

        self.assertEqual(result.kv_hit_pages, 3)
        self.assertEqual(result.extra_pool_hit_pages, {pool_transfer.PoolName.MAMBA: 1})


if __name__ == "__main__":
    unittest.main()
