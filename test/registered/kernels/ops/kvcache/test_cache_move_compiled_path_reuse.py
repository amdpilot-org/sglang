import time

import pytest
import torch

from sglang.kernels.ops.kvcache.cache_move import (
    copy_all_layer_kv_cache_func,
    copy_all_layer_kv_cache_tiled,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=2, stage="base-b", runner_config="1-gpu-large")
register_amd_ci(est_time=2, stage="jit-kernel-unit", runner_config="amd")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="cache-move compiled-path reuse tests require CUDA or ROCm.",
)

DEVICE = "cuda"
DTYPE = torch.bfloat16
SENTINEL = -987654.0
NUM_LOCS_SEQUENCE = [3, 5, 7, 9]
NUM_LOCS_UPPER = 16
KV_COPY_CONFIG = {
    "bytes_per_tile": 128,
    "byte_tiles": 1,
    "num_warps": 4,
}


def _make_case():
    buffers = [
        torch.randn(16, 16, dtype=DTYPE, device=DEVICE) for _ in range(2)
    ]
    for buffer in buffers:
        buffer[[0, -1]] = SENTINEL

    data_ptrs = torch.tensor(
        [buffer.data_ptr() for buffer in buffers],
        dtype=torch.uint64,
        device=DEVICE,
    )
    strides = torch.full(
        (2,),
        buffers[0][0].numel() * DTYPE.itemsize,
        dtype=torch.int64,
        device=DEVICE,
    )
    return buffers, data_ptrs, strides


@pytest.mark.parametrize(
    "invalid_variant",
    [
        "data_ptrs_int64",
        "strides_int32",
        "tgt_loc_int32",
        "src_loc_int32",
    ],
)
def test_cache_move_rejects_unsupported_gpu_dtypes(invalid_variant: str) -> None:
    buffers, data_ptrs, strides = _make_case()
    src_loc = torch.tensor([1, 2, 3], dtype=torch.int64, device=DEVICE)
    tgt_loc = torch.tensor([2, 3, 4], dtype=torch.int64, device=DEVICE)

    if invalid_variant == "data_ptrs_int64":
        data_ptrs = data_ptrs.to(torch.int64)
        expected_message = "data_ptrs must have dtype torch.uint64"
    elif invalid_variant == "strides_int32":
        strides = strides.to(torch.int32)
        expected_message = "strides must have dtype torch.int64"
    elif invalid_variant == "tgt_loc_int32":
        tgt_loc = tgt_loc.to(torch.int32)
        expected_message = "tgt_loc and src_loc must have dtype torch.int64"
    else:
        src_loc = src_loc.to(torch.int32)
        expected_message = "tgt_loc and src_loc must have dtype torch.int64"

    with pytest.raises(TypeError, match=expected_message):
        copy_all_layer_kv_cache_func(
            data_ptrs,
            strides,
            tgt_loc,
            src_loc,
            len(src_loc),
            NUM_LOCS_UPPER,
            KV_COPY_CONFIG,
        )


def test_cache_move_cold_warm_compiled_path_reuse() -> None:
    buffers, data_ptrs, strides = _make_case()
    original = [buffer.cpu().clone() for buffer in buffers]
    pointer_tensor_address = data_ptrs.data_ptr()
    cache_before = len(copy_all_layer_kv_cache_tiled.device_caches[0][0])
    timings_ms = {}

    for num_locs in NUM_LOCS_SEQUENCE:
        for buffer, reference in zip(buffers, original):
            buffer.copy_(reference)

        src_loc = torch.arange(
            1, num_locs + 1, dtype=torch.int64, device=DEVICE
        )
        tgt_loc = torch.arange(
            2, num_locs + 2, dtype=torch.int64, device=DEVICE
        )
        torch.cuda.synchronize()

        start_ns = time.perf_counter_ns()
        copy_all_layer_kv_cache_func(
            data_ptrs,
            strides,
            tgt_loc,
            src_loc,
            num_locs,
            NUM_LOCS_UPPER,
            KV_COPY_CONFIG,
        )
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000
        timings_ms[num_locs] = elapsed_ms

        for buffer, reference in zip(buffers, original):
            expected = reference.clone()
            expected.index_copy_(
                0,
                tgt_loc.cpu(),
                reference.index_select(0, src_loc.cpu()),
            )
            assert torch.equal(buffer.cpu(), expected)
            assert buffer[[0, -1]].cpu().equal(
                torch.full((2, 16), SENTINEL, dtype=DTYPE)
            )

    cache_after = len(copy_all_layer_kv_cache_tiled.device_caches[0][0])
    assert cache_after - cache_before == 1
    assert data_ptrs.data_ptr() == pointer_tensor_address
    assert data_ptrs.tolist() == [buffer.data_ptr() for buffer in buffers]
    print(f"cache_move_reuse_timings_ms={timings_ms}")
