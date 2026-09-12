import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.srt.mem_cache.common import retraction_backup
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.mem_cache.memory_pool import KVCache
from sglang.test.ci.ci_register import register_cpu_ci


register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestRetractionBackupFailure(unittest.TestCase):
    def setUp(self):
        self.req_to_token_pool = Mock()
        self.allocator = Mock()
        self.tree_cache = Mock()

    def _req(self, side_effect=None):
        req = SimpleNamespace(rid="dsv4-request", kv=SimpleNamespace())
        req.offload_kv_cache = Mock(side_effect=side_effect)
        return req

    def test_dsv4_pool_uses_unsupported_cpu_copy_stub(self):
        self.assertIs(DeepSeekV4TokenToKVPool.get_cpu_copy, KVCache.get_cpu_copy)

    def test_unsupported_cpu_copy_declines_backup(self):
        req = self._req(NotImplementedError("DeepSeek-V4 CPU copy unsupported"))

        self.assertFalse(
            retraction_backup(
                req,
                self.tree_cache,
                self.req_to_token_pool,
                self.allocator,
                "cpu_tensor",
            )
        )

    def test_supported_cpu_copy_saves_backup(self):
        req = self._req()

        self.assertTrue(
            retraction_backup(
                req,
                self.tree_cache,
                self.req_to_token_pool,
                self.allocator,
                "cpu_tensor",
            )
        )
        req.offload_kv_cache.assert_called_once_with(
            self.req_to_token_pool, self.allocator
        )

    def test_unexpected_cpu_copy_error_is_not_hidden(self):
        req = self._req(RuntimeError("device copy failed"))

        with self.assertRaisesRegex(RuntimeError, "device copy failed"):
            retraction_backup(
                req,
                self.tree_cache,
                self.req_to_token_pool,
                self.allocator,
                "cpu_tensor",
            )


if __name__ == "__main__":
    unittest.main()
