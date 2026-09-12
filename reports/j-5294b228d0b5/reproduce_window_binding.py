"""Reproduce the reviewed allocator accessor binding failure with a real rebuild."""

import ctypes
import os
import tempfile

import torch
import torch.distributed as dist

from sglang.srt.distributed.device_communicators import pynccl_allocator as allocator


def main() -> None:
    dist.init_process_group(
        backend="gloo",
        init_method="tcp://127.0.0.1:29653",
        rank=0,
        world_size=1,
    )
    try:
        pool = allocator.get_nccl_mem_pool()
        del pool
        # get_nccl_mem_pool uses tempfile.gettempdir()/symm_allocator.
        library_path = os.path.join(
            tempfile.gettempdir(), "symm_allocator", "nccl_allocator.so"
        )
        native = ctypes.CDLL(library_path)
        print(f"native_library={library_path}")
        print(
            "native_get_windows_symbol=",
            hasattr(native, "nccl_allocator_get_windows_for_comm"),
        )
        print(
            "native_clear_windows_symbol=",
            hasattr(native, "nccl_allocator_clear_windows_for_comm"),
        )
        print("module_get_windows_func=", allocator._get_windows_func)
        print("module_clear_windows_func=", allocator._clear_windows_func)
        print("collected_windows=", allocator._collect_windows_for_comm(0x1234))
        assert allocator._get_windows_func is not None
        assert allocator._clear_windows_func is not None
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
