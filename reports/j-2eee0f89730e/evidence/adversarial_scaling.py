import inspect
import statistics
import time

import torch

from sglang.kernels.ops.attention.dsa.dequant_k_cache import (
    dequantize_k_cache_paged,
    dequantize_k_cache_paged_selective,
)

torch.manual_seed(20260912)
rows = 131072
topk_width = 2048
cache = torch.zeros((rows, 1, 656), device="cuda", dtype=torch.float8_e4m3fn)
page_table = torch.randperm(rows, device="cuda", dtype=torch.int32)


def timed(fn):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start_alloc = torch.cuda.memory_allocated()
    start = time.perf_counter()
    value = fn()
    torch.cuda.synchronize()
    return (
        (time.perf_counter() - start) * 1000,
        torch.cuda.max_memory_allocated() - start_alloc,
        value,
    )


def measured(fn, repeats=7):
    warm = fn()
    torch.cuda.synchronize()
    del warm
    samples = [timed(fn) for _ in range(repeats)]
    return (
        statistics.median(x[0] for x in samples),
        max(x[1] for x in samples),
        samples[-1][2],
    )


print(f"module={inspect.getfile(dequantize_k_cache_paged_selective)}")
print(f"device={torch.cuda.get_device_name(0)}")
full_ms, full_peak, full = measured(lambda: dequantize_k_cache_paged(cache, page_table))
print(f"full_ms={full_ms:.3f} full_peak_bytes={full_peak} full_rows={full.shape[0]}")
del full

for query_rows in (8, 64, 256, 1024):
    topk = torch.randint(
        0, rows, (query_rows, topk_width), device="cuda", dtype=torch.int32
    )
    ms, peak, result = measured(
        lambda: dequantize_k_cache_paged_selective(cache, page_table, topk)
    )
    kv, remapped, used = result
    print(
        f"query_rows={query_rows} entries={topk.numel()} selective_used={used} "
        f"output_rows={kv.shape[0]} ms={ms:.3f} peak_bytes={peak} "
        f"vs_full={ms / full_ms:.3f}x"
    )
    del topk, kv, remapped, result
