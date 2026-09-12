import unittest
from unittest.mock import MagicMock, patch

from sglang.srt.managers.schedule_batch import retract_all
from sglang.test.ci.ci_register import register_cpu_ci


register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestRetractAllCacheRelease(unittest.TestCase):
    def _retract(self, tree_cache, reqs=None):
        retract_all(
            reqs=[MagicMock(), MagicMock()] if reqs is None else reqs,
            req_to_token_pool=MagicMock(),
            token_to_kv_pool_allocator=MagicMock(),
            tree_cache=tree_cache,
            hisparse_coordinator=None,
            offload_kv=False,
        )

    @patch("sglang.srt.managers.schedule_batch.release_req")
    def test_retracts_every_request(self, release_req):
        tree_cache = MagicMock()

        self._retract(tree_cache)

        self.assertEqual(release_req.call_count, 2)

    @patch("sglang.srt.managers.schedule_batch.release_req")
    def test_empty_request_list_is_a_noop(self, release_req):
        tree_cache = MagicMock()

        self._retract(tree_cache, reqs=[])

        release_req.assert_not_called()


if __name__ == "__main__":
    unittest.main()
