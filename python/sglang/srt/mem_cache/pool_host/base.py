from __future__ import annotations

import abc
import fcntl
import logging
import os
import tempfile
import threading
from functools import wraps
from typing import Optional

import psutil
import torch

from sglang.srt.distributed.parallel_state import get_world_group
from sglang.srt.mem_cache.memory_pool import KVCache
from sglang.srt.mem_cache.pool_host.common import (
    _cuda_host_unregister,
    get_allocator_from_storage,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import is_cuda, is_hip

logger = logging.getLogger(__name__)

_is_cuda = is_cuda()
_is_hip = is_hip()

# Host RAM to leave free when sizing HiCache pools (OS, other processes).
HICACHE_HOST_MEMORY_RESERVE_BYTES: int = 10 * (1024**3)

_WRITE_BACK_STAGING_PAGE_CHUNK = 64


def ranks_per_host() -> int:
    """Number of ranks of this job running on the same machine as this one.

    Derived as world_size // nnodes: the launcher slices ranks uniformly
    across nodes (resolution asserts divisibility), so no hostname collective
    is needed — a collective here would have to be issued the same number of
    times on every rank, and ranks on different pipeline stages build
    different numbers of host pools.
    """
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return 1
    try:
        world_group = get_world_group()
    except AssertionError:
        return 1
    if world_group.world_size == 1:
        return 1
    return max(world_group.world_size // get_parallel().nnodes, 1)


def host_memory_sync_group() -> Optional[torch.distributed.ProcessGroup]:
    """Return the world CPU group used for the initial host-memory snapshot.

    Every rank that enables HiCache constructs a primary host pool, so the first
    sizing check is common even when later pipeline stages construct different
    sidecar pools. Synchronizing that check across the whole job also covers
    multiple TP or in-job DP groups co-located on one host.
    """
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return None
    try:
        world_group = get_world_group()
    except AssertionError:
        return None
    if world_group.world_size <= 1:
        return None
    return world_group.cpu_group


_initial_host_memory_available_bytes: Optional[int] = None
_reserved_host_memory_bytes = 0
_host_memory_budget_lock = threading.Lock()
_HOST_MEMORY_ALLOCATION_LOCK_PATH = os.path.join(
    tempfile.gettempdir(), "sglang-hicache-host-memory.lock"
)


class HostMemoryAllocationLock:
    """Serialize a host-memory check with the allocation it protects."""

    def __init__(self):
        self._file = None

    def acquire(self) -> int:
        _host_memory_budget_lock.acquire()
        try:
            self._file = open(_HOST_MEMORY_ALLOCATION_LOCK_PATH, "a+")
            fcntl.flock(self._file.fileno(), fcntl.LOCK_EX)
            return max(
                psutil.virtual_memory().available - HICACHE_HOST_MEMORY_RESERVE_BYTES,
                0,
            )
        except BaseException:
            _host_memory_budget_lock.release()
            raise

    def release(self):
        if self._file is None:
            return
        try:
            fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
            self._file.close()
            self._file = None
        finally:
            _host_memory_budget_lock.release()

    def __enter__(self) -> int:
        return self.acquire()

    def __exit__(self, exc_type, exc, traceback):
        self.release()

    def __del__(self):
        self.release()


def host_memory_allocation_lock() -> HostMemoryAllocationLock:
    """Coordinate HiCache allocations across ranks and separate local jobs.

    The lock must cover both the live-memory check and the allocation. Merely
    locking the check leaves a race where another process can pass against the
    same snapshot before either process has made its allocation visible.
    """
    return HostMemoryAllocationLock()


def host_memory_budget_bytes(requested_bytes: int = 0) -> int:
    """Remaining per-rank host RAM before an optional pool reservation.

    Take one job-wide synchronized snapshot before any rank allocates its first
    pool, then keep that baseline for later pools. This makes the result
    independent of allocation timing across TP, PP, and in-job DP groups while
    avoiding collectives for sidecar pools that not every rank constructs.

    Accepted requests are accumulated locally because the baseline no longer
    falls as this process allocates its earlier pools. The equal per-rank split
    is intentionally preserved from the original guard.
    """
    global _initial_host_memory_available_bytes, _reserved_host_memory_bytes

    with _host_memory_budget_lock:
        if _initial_host_memory_available_bytes is None:
            free = psutil.virtual_memory().available
            sync_group = host_memory_sync_group()
            if sync_group is not None:
                reading = torch.tensor(free, dtype=torch.int64)
                torch.distributed.all_reduce(
                    reading, op=torch.distributed.ReduceOp.MIN, group=sync_group
                )
                free = int(reading.item())
            _initial_host_memory_available_bytes = free

        total_budget = (
            _initial_host_memory_available_bytes - HICACHE_HOST_MEMORY_RESERVE_BYTES
        ) // ranks_per_host()
        remaining_budget = total_budget - _reserved_host_memory_bytes
        if requested_bytes <= remaining_budget:
            _reserved_host_memory_bytes += requested_bytes
        return remaining_budget


def sync_fixed_hicache_size(size: int, host_size: int) -> int:
    """Sync fixed-size HiCache token capacity across PP ranks.

    A fixed --hicache-size is specified in GB, but each PP stage may have a
    different bytes/token because it owns different layers. Use the global
    minimum token capacity within the PP group so all stages expose the same
    host-cache capacity.
    Ratio-based sizing already derives from the synced device pool size.
    """
    if host_size <= 0 or not torch.distributed.is_available():
        return size

    if not torch.distributed.is_initialized():
        return size

    try:
        from sglang.srt.distributed.parallel_state import get_pp_group

        pp_group = get_pp_group()
    except AssertionError:
        return size

    if pp_group.world_size <= 1:
        return size

    tensor = torch.tensor(size, dtype=torch.int64)
    torch.distributed.all_reduce(
        tensor,
        op=torch.distributed.ReduceOp.MIN,
        group=pp_group.cpu_group,
    )
    synced_size = int(tensor.item())

    if synced_size != size:
        logger.info(
            "Sync fixed-size HiCache host token capacity from %d to %d.",
            size,
            synced_size,
        )
    return synced_size


def synchronized(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        with self.lock:
            return func(self, *args, **kwargs)

    return wrapper


class HostKVCache(abc.ABC):
    dcp_size = 1
    dcp_rank = 0

    def __init__(
        self,
        device_pool: KVCache,
        host_to_device_ratio: float,
        host_size: int,
        page_size: int,
        layout: str,
        pin_memory: bool,
        device: str,
        allocator_type: str = "default",
        dcp_size: int = 1,
        dcp_rank: int = 0,
        *,
        pool_label: str = "kv",
    ):
        self.device_pool = device_pool
        self.pool_label = pool_label
        # page_size arrives widened (x dcp_size); size/page_size/page_num are physical.
        self.dcp_size = dcp_size
        self.dcp_rank = dcp_rank
        assert page_size % dcp_size == 0, (
            f"HiCache host pool page_size ({page_size}) must be a multiple of "
            f"dcp_size ({dcp_size}); expected the widened page from the DCP "
            "paged allocator."
        )
        self.page_size = page_size // dcp_size
        self.layout = layout
        self.pin_memory = pin_memory
        self.device = device
        self.allocator = get_allocator_from_storage(allocator_type)
        self.can_use_write_back_jit = False

        self.dtype = device_pool.store_dtype
        self.size_per_token = self.get_size_per_token()
        if host_size > 0:
            self.size = sync_fixed_hicache_size(
                int(host_size * 1e9 // self.size_per_token), host_size
            )
        else:
            self.size = int(device_pool.size * host_to_device_ratio)
        # Align up the host memory pool size to the page size
        self.page_num = self.size // self.page_size + 1
        self.size = self.page_num * self.page_size
        self.start_layer = device_pool.start_layer
        self.end_layer = device_pool.end_layer

        if self.size <= device_pool.size:
            logger.warning(
                "HiCache %s host pool (%d tokens) is smaller than the device pool (%d tokens);"
                "L2 cache effectiveness is reduced."
                "Consider increasing --hicache-ratio (or --hicache-size) for higher L2 cache hit rate.",
                pool_label,
                self.size,
                device_pool.size,
            )

        requested_bytes = self.size * self.size_per_token
        with host_memory_allocation_lock() as available_bytes:
            if requested_bytes > available_bytes:
                raise ValueError(
                    f"Not enough host memory available. Requesting "
                    f"{requested_bytes / 1e9:.2f} GB but only have "
                    f"{available_bytes / 1e9:.2f} GB free. Please reduce the "
                    f"size of the hierarchical cache."
                )
            draft_layer_num = self.layer_num - self.target_layer_num
            if draft_layer_num > 0:
                logger.info(
                    "Allocating %s hierarchical KV host pool: %d tokens, "
                    "%.2f GB host memory, packed MTP KV layers: "
                    "target_layers=%d, draft_layers=%d, total_layers=%d.",
                    pool_label,
                    self.size,
                    requested_bytes / 1e9,
                    self.target_layer_num,
                    draft_layer_num,
                    self.layer_num,
                )
            else:
                logger.info(
                    "Allocating %s hierarchical KV host pool: %d tokens, %.2f GB host memory.",
                    pool_label,
                    self.size,
                    requested_bytes / 1e9,
                )

            self.kv_buffer = self.init_kv_buffer()
        self.fd = getattr(self.allocator, "fd", None)

        # A lock for synchronized operations on memory allocation and state transitions.
        self.lock = threading.RLock()
        self.clear()

    def destroy(self):
        """Unregister pinned host buffers in userspace before process exit.

        Large cudaHostRegister'd buffers are otherwise unpinned by the kernel
        during SIGKILL reclaim, which can stall teardown in uninterruptible
        sleep for tens of seconds. Idempotent. (Only the host_register path
        needs this; npu/musa pin_memory buffers are freed by torch.)
        """
        if getattr(self, "_destroyed", False):
            return
        self._destroyed = True
        buffers = getattr(self, "kv_buffer", None)
        if buffers is not None and self.pin_memory and (_is_cuda or _is_hip):
            if not isinstance(buffers, (list, tuple)):
                buffers = [buffers]
            for buf in buffers:
                if buf is not None:
                    _cuda_host_unregister(buf)
        self.kv_buffer = None

    @abc.abstractmethod
    def get_size_per_token(self):
        raise NotImplementedError()

    def _is_device_layer_sharded(self, device_pool=None) -> bool:
        device_pool = device_pool or self.device_pool
        return bool(device_pool.layer_shard_enabled)

    def _device_owned_layer_range(self, device_pool=None) -> tuple[int, int]:
        """Contiguous ``[start, end)`` local device layers this rank stores.

        ``(0, layer_num)`` when the device pool is not layer-sharded.
        """
        device_pool = device_pool or self.device_pool
        if not self._is_device_layer_sharded(device_pool):
            return 0, device_pool.layer_num
        return device_pool._owned_local_layer_range()

    def _effective_host_layer_num(self, device_pool=None) -> int:
        """Number of layers the host pool allocates for this rank."""
        device_pool = device_pool or self.device_pool
        if not self._is_device_layer_sharded(device_pool):
            return device_pool.layer_num
        shard_size = device_pool.layer_shard_size
        return (device_pool.layer_num + shard_size - 1) // shard_size

    def _is_device_layer_owned(self, device_pool, layer_id: int) -> bool:
        start, end = self._device_owned_layer_range(device_pool)
        return start <= layer_id < end

    def _host_layer_index(self, layer_id: int, device_pool=None) -> int:
        start, _ = self._device_owned_layer_range(device_pool)
        return layer_id - start

    def _owned_device_layer_ids(self, device_pool) -> list[int]:
        start, end = self._device_owned_layer_range(device_pool)
        return list(range(start, end))

    @abc.abstractmethod
    def init_kv_buffer(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def load_to_device_per_layer(
        self,
        device_pool,
        host_indices,
        device_indices,
        layer_id,
        io_backend,
        *,
        is_draft: bool = False,
    ) -> None:
        """
        Load KV data from the host memory pool to the device memory pool for a specific layer.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ) -> None:
        """
        Backup KV data from the device memory pool to the host memory pool for all layers.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_data_page(self, index, flat: bool = True) -> torch.Tensor:
        """
        Get a flat data page from the host memory pool.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_dummy_flat_data_page(self) -> torch.Tensor:
        """
        Get a dummy flat data page from the host memory pool.
        This is used for prefetching or initializing empty pages.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def set_from_flat_data_page(self, index: int, data_page: torch.Tensor) -> None:
        """
        Set a flat data page to the host memory pool.
        """
        raise NotImplementedError()

    def is_stride_page_aligned(self, page_size_bytes: int = 4096) -> bool:
        """Return True if per-page strides are multiples of *page_size_bytes*.

        Subclasses should override this with a layout-specific stride formula.
        This base implementation logs a warning and returns False (safe default).
        """
        logger.warning(
            "%s does not implement is_stride_page_aligned(); assuming not aligned. "
            "O_DIRECT with a file-based NIXL backend will fall back to copy mode for this pool.",
            type(self).__name__,
        )
        return False

    @synchronized
    def clear(self):
        # Initialize memory states and tracking structures.
        self.mem_state = torch.zeros(
            (self.logical_size,), dtype=torch.uint8, device=self.device
        )
        self.free_slots = torch.arange(self.logical_size, dtype=torch.int64)
        # Keep freed chunks aside and consume them lazily from alloc() to avoid
        # concatenating a large free-list on every host-pool free.
        self.release_slots = []
        self.num_release_slots = 0
        # Per-slot flag used to detect double-free.
        # slot_used[k] is true if slot k is allocated.
        self.slot_used = torch.zeros(self.logical_size, dtype=torch.bool)

    def available_size(self):
        return len(self.free_slots) + self.num_release_slots

    def _merge_release_slots(self):
        if self.num_release_slots == 0:
            return

        if len(self.free_slots) == 0 and len(self.release_slots) == 1:
            self.free_slots = self.release_slots[0]
        else:
            self.free_slots = torch.cat([self.free_slots, *self.release_slots])

        self.release_slots = []
        self.num_release_slots = 0

    @property
    def logical_size(self) -> int:
        """Slots the radix/controller layer sees: dcp_size of them share a row."""
        return self.size * self.dcp_size

    @property
    def logical_page_size(self) -> int:
        """Page size in that same logical space (the widened DCP page)."""
        return self.page_size * self.dcp_size

    def maybe_dcp_kernel_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Transfer kernels index per-rank rows; callers hold widened logical slots.

        Keep this rank's slots (% dcp_size == dcp_rank), then collapse (// dcp_size).
        """
        if self.dcp_size == 1:
            return indices
        assert indices.numel() % self.dcp_size == 0, (
            "HiCache DCP translation expects runs of whole widened pages; got "
            f"{indices.numel()} logical slots with dcp_size={self.dcp_size}."
        )
        return indices[self.dcp_rank :: self.dcp_size] // self.dcp_size

    @synchronized
    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        assert need_size % self.logical_page_size == 0, (
            "The requested size should be a multiple of the page size."
        )
        if need_size > self.available_size():
            return None

        if need_size > len(self.free_slots):
            self._merge_release_slots()

        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]

        assert not self.slot_used[select_index].any(), (
            f"Double-alloc detected: slots already allocated: "
            f"{select_index[self.slot_used[select_index]].tolist()}."
        )
        self.slot_used[select_index] = True

        return select_index

    @synchronized
    def free(self, indices: torch.Tensor) -> int:
        indices_cpu = indices.cpu()
        if indices_cpu.numel() == 0:
            return 0

        assert self.slot_used[indices_cpu].all(), (
            f"Double-free detected: slots not currently allocated: "
            f"{indices_cpu[~self.slot_used[indices_cpu]].tolist()}."
        )
        self.slot_used[indices_cpu] = False
        self.release_slots.append(indices_cpu)
        self.num_release_slots += len(indices_cpu)
        return len(indices)
