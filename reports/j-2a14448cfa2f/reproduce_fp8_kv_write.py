"""Reproduce issue #30815's KV-write overhead through MHATokenToKVPool.

This is intentionally a pool-level fixture, not an end-to-end model claim.  It
profiles the actual set_kv_buffer implementation and checks the stored FP8 bytes
against an independent PyTorch expression.
"""

import json
import statistics
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool


DEVICE = "cuda"
HEADS = 8
HEAD_DIM = 128
ROWS = 65
WARMUP = 25
REPEATS = 200


def make_pool(dtype: torch.dtype) -> MHATokenToKVPool:
    return MHATokenToKVPool(
        size=ROWS - 1,
        page_size=1,
        dtype=dtype,
        head_num=HEADS,
        head_dim=HEAD_DIM,
        layer_num=1,
        device=DEVICE,
        enable_memory_saver=False,
        enable_alt_stream=False,
    )


def timed_us(call) -> list[float]:
    for _ in range(WARMUP):
        call()
    torch.cuda.synchronize()
    samples = []
    for _ in range(REPEATS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        call()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0)
    return samples


def profile_kernel_names(call) -> list[str]:
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(activities=activities) as prof:
        call()
        torch.cuda.synchronize()
    names = []
    for event in prof.events():
        if event.device_type != torch.autograd.DeviceType.CPU:
            names.append(event.name)
    return names


def main() -> None:
    torch.manual_seed(30815)
    fp8 = torch.float8_e4m3fnuz
    layer = SimpleNamespace(layer_id=0)
    loc = torch.tensor([1], dtype=torch.int64, device=DEVICE)
    k_src = torch.randn((1, HEADS, HEAD_DIM), dtype=torch.bfloat16, device=DEVICE)
    v_src = torch.randn((1, HEADS, HEAD_DIM), dtype=torch.bfloat16, device=DEVICE)
    k_scale = torch.tensor([0.5], dtype=torch.float32, device=DEVICE)
    v_scale = torch.tensor([0.25], dtype=torch.float32, device=DEVICE)

    bf16_pool = make_pool(torch.bfloat16)
    fp8_pool = make_pool(fp8)

    def bf16_write() -> None:
        bf16_pool.set_kv_buffer(layer, loc, k_src.clone(), v_src.clone())

    def fp8_write() -> None:
        # Clone because the current implementation divides inputs in-place.
        fp8_pool.set_kv_buffer(
            layer,
            loc,
            k_src.clone(),
            v_src.clone(),
            k_scale=k_scale,
            v_scale=v_scale,
        )

    bf16_us = timed_us(bf16_write)
    fp8_us = timed_us(fp8_write)
    bf16_kernels = profile_kernel_names(bf16_write)
    fp8_kernels = profile_kernel_names(fp8_write)

    fp8_write()
    expected_k = (k_src / k_scale).to(fp8)
    expected_v = (v_src / v_scale).to(fp8)
    # The physical HIP allocation is byte-backed; public accessors restore the
    # logical FP8 view consumed by attention kernels.
    actual_k = fp8_pool.get_key_buffer(0)[loc]
    actual_v = fp8_pool.get_value_buffer(0)[loc]

    result = {
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "device": torch.cuda.get_device_name(0),
        "arch": torch.cuda.get_device_properties(0).gcnArchName,
        "shape": [1, HEADS, HEAD_DIM],
        "bf16_median_us": statistics.median(bf16_us),
        "fp8_scaled_median_us": statistics.median(fp8_us),
        "ratio": statistics.median(fp8_us) / statistics.median(bf16_us),
        "bf16_device_events": bf16_kernels,
        "fp8_scaled_device_events": fp8_kernels,
        "bf16_device_event_count": len(bf16_kernels),
        "fp8_scaled_device_event_count": len(fp8_kernels),
        "k_bit_exact": bool(torch.equal(actual_k, expected_k)),
        "v_bit_exact": bool(torch.equal(actual_v, expected_v)),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
