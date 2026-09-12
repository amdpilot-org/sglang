import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.managers.schedule_batch import release_req
from sglang.srt.mem_cache.common import retraction_backup
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.mem_cache.memory_pool import KVCache
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cpu_ci


register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestRetractionBackupFallback(unittest.TestCase):
    def setUp(self):
        self.req_to_token_pool = Mock()
        self.allocator = Mock()
        self.tree_cache = Mock()

    @staticmethod
    def _req(side_effect=None):
        req = SimpleNamespace(
            rid="dsv4-request",
            kv=SimpleNamespace(),
            finished=Mock(return_value=False),
            reset_for_retract=Mock(),
        )
        req.offload_kv_cache = Mock(side_effect=side_effect)
        return req

    def _set_decode_args(self, *, offload_enabled: bool):
        args = ServerArgs(
            model_path="dummy",
            disaggregation_mode="decode",
            disaggregation_decode_enable_offload_kvcache=offload_enabled,
            disaggregation_decode_retraction_backup="cpu_tensor",
        )
        set_global_server_args_for_scheduler(args)

    def test_dsv4_pool_uses_unsupported_cpu_copy_stub(self):
        self.assertIs(DeepSeekV4TokenToKVPool.get_cpu_copy, KVCache.get_cpu_copy)

    def test_unsupported_cpu_copy_requests_rebootstrap(self):
        req = self._req(NotImplementedError("DeepSeek-V4 CPU copy unsupported"))

        self.assertIsNone(
            retraction_backup(
                req,
                self.tree_cache,
                self.req_to_token_pool,
                self.allocator,
                "cpu_tensor",
            )
        )

    @patch("sglang.srt.managers.schedule_batch.evict_from_tree_cache")
    @patch("sglang.srt.managers.schedule_batch.release_kv_cache")
    def test_disabled_offload_skips_cpu_copy_and_requests_rebootstrap(
        self, release_kv_cache, evict_from_tree_cache
    ):
        self._set_decode_args(offload_enabled=False)
        req = self._req(NotImplementedError("must not be called"))

        result = release_req(
            req=req,
            remaing_req_count=0,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.allocator,
            tree_cache=self.tree_cache,
            hisparse_coordinator=None,
        )

        req.offload_kv_cache.assert_not_called()
        self.assertIsNone(result)
        release_kv_cache.assert_called_once_with(
            req, self.tree_cache, is_insert=False
        )
        req.reset_for_retract.assert_called_once_with()
        evict_from_tree_cache.assert_called_once()

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
