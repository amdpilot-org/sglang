import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestMlaRetractionDcp(unittest.TestCase):
    @staticmethod
    def _pool(values: torch.Tensor, chunk_size: int = 2) -> MLATokenToKVPool:
        pool = MLATokenToKVPool.__new__(MLATokenToKVPool)
        pool.layer_num = 1
        pool.kv_buffer = [values.clone()]
        pool.cpu_offloading_chunk_size = chunk_size
        return pool

    @staticmethod
    def _parallel(dcp_size: int, dcp_rank: int):
        return SimpleNamespace(
            attn_dcp_size=dcp_size,
            attn_dcp_rank=dcp_rank,
        )

    def test_backup_selects_owned_rows_and_localizes_in_chunks(self):
        pool = self._pool(torch.arange(10).reshape(10, 1), chunk_size=2)
        logical = torch.tensor([8, 9, 10, 11, 20, 21, 22, 23])

        with (
            mock.patch(
                "sglang.srt.mem_cache.memory_pool.get_parallel",
                return_value=self._parallel(4, 2),
            ),
            mock.patch("sglang.srt.mem_cache.memory_pool.current_platform.synchronize"),
        ):
            backup = pool.get_cpu_copy(logical)

        self.assertEqual(len(backup[0]), 1)
        torch.testing.assert_close(backup[0][0], torch.tensor([[2], [5]]))

    def test_restore_uses_new_logical_locations(self):
        source = self._pool(torch.arange(10).reshape(10, 1))
        destination = self._pool(torch.full((10, 1), -1))

        with (
            mock.patch(
                "sglang.srt.mem_cache.memory_pool.get_parallel",
                return_value=self._parallel(4, 1),
            ),
            mock.patch("sglang.srt.mem_cache.memory_pool.current_platform.synchronize"),
        ):
            backup = source.get_cpu_copy(torch.tensor([4, 5, 6, 7, 16, 17, 18, 19]))
            destination.load_cpu_copy(
                backup, torch.tensor([24, 25, 26, 27, 32, 33, 34, 35])
            )

        torch.testing.assert_close(
            destination.kv_buffer[0][[6, 8]], torch.tensor([[1], [4]])
        )
        self.assertTrue(bool((destination.kv_buffer[0][[1, 4]] == -1).all()))

    def test_dcp_one_keeps_all_indices(self):
        values = torch.arange(8).reshape(8, 1)
        pool = self._pool(values, chunk_size=3)

        with (
            mock.patch(
                "sglang.srt.mem_cache.memory_pool.get_parallel",
                return_value=self._parallel(1, 0),
            ),
            mock.patch("sglang.srt.mem_cache.memory_pool.current_platform.synchronize"),
        ):
            backup = pool.get_cpu_copy(torch.tensor([1, 3, 6]))

        torch.testing.assert_close(torch.cat(backup[0]), values[[1, 3, 6]])


if __name__ == "__main__":
    unittest.main()
