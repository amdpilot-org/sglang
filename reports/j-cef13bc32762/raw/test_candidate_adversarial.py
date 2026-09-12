import tempfile

import torch

from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.memory_pool_host import LogicalHostPool
from sglang.srt.mem_cache.storage.hf3fs.mini_3fs_metadata_server import Hf3fsLocalMetadataClient
from sglang.srt.mem_cache.storage.hf3fs.storage_hf3fs import HiCacheHF3FS


class PhysicalSidePool:
    def __init__(self):
        self.page_size = 2
        self.dtype = torch.uint8
        self.layout = "page_first"
        self.kv_buffer = torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23]], dtype=torch.uint8)

    def get_ksize_per_token(self):
        return 2

    def get_data_page(self, index, flat=True):
        return self.kv_buffer[index // self.page_size]

    def get_dummy_flat_data_page(self):
        return torch.empty(4, dtype=torch.uint8)

    def set_from_flat_data_page(self, index, value):
        self.kv_buffer[index // self.page_size].copy_(value)


def test_logical_anchor_gates_and_round_trips_physical_v2_pool():
    logical = LogicalHostPool(size=4, page_size=2, layout="page_first_direct")
    physical = PhysicalSidePool()
    metadata = Hf3fsLocalMetadataClient()
    with tempfile.TemporaryDirectory() as td:
        backend = HiCacheHF3FS(
            rank=0, file_path=f"{td}/anchor.bin", file_size=64, numjobs=1,
            bytes_per_page=1, entries=4, client_timeout=1, dtype=torch.uint8,
            metadata_client=metadata, use_mock_client=True,
        )
        try:
            backend.register_mem_pool_host(logical)
            backend.register_mem_host_pool_v2(physical, PoolName.DEEPSEEK_V4_C4)
            indices = torch.arange(4, dtype=torch.int64)
            keys = ["p0", "p1"]
            transfer = PoolTransfer(name=PoolName.DEEPSEEK_V4_C4, host_indices=indices, keys=keys)

            assert backend.batch_set_v1(keys, indices) == [True, True]
            # Only write the first physical component: v2 existence must truncate to one.
            first = PoolTransfer(name=PoolName.DEEPSEEK_V4_C4, host_indices=indices[:2], keys=keys[:1])
            assert backend.batch_set_v2([first])[PoolName.DEEPSEEK_V4_C4] == [True]
            hit = backend.batch_exists_v2(keys, [transfer])
            assert hit.kv_hit_pages == 1

            # Write the second component and prove actual bytes restore, not merely metadata.
            second = PoolTransfer(name=PoolName.DEEPSEEK_V4_C4, host_indices=indices[2:], keys=keys[1:])
            assert backend.batch_set_v2([second])[PoolName.DEEPSEEK_V4_C4] == [True]
            physical.kv_buffer.zero_()
            hit = backend.batch_exists_v2(keys, [transfer])
            assert hit.kv_hit_pages == 2
            assert backend.batch_get_v1(keys, indices) == [True, True]
            assert backend.batch_get_v2([transfer])[PoolName.DEEPSEEK_V4_C4] == [True, True]
            assert physical.kv_buffer.tolist() == [[10, 11, 12, 13], [20, 21, 22, 23]]
        finally:
            backend.close()


def test_zero_sized_named_side_pool_still_fails_explicitly():
    logical = LogicalHostPool(size=2, page_size=2, layout="page_first_direct")
    physical = PhysicalSidePool()
    physical.get_ksize_per_token = lambda: 0
    with tempfile.TemporaryDirectory() as td:
        backend = HiCacheHF3FS(
            rank=0, file_path=f"{td}/anchor.bin", file_size=8, numjobs=1,
            bytes_per_page=1, entries=1, client_timeout=1, dtype=torch.uint8,
            metadata_client=Hf3fsLocalMetadataClient(), use_mock_client=True,
        )
        try:
            backend.register_mem_pool_host(logical)
            try:
                backend.register_mem_host_pool_v2(physical, PoolName.DEEPSEEK_V4_C4)
            except ZeroDivisionError:
                pass
            else:
                raise AssertionError("zero-sized physical side pool was silently accepted")
        finally:
            backend.close()
