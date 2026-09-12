import inspect
import statistics
import time

import torch

from sglang.kernels.ops.attention.dsa.dequant_k_cache import (
    dequantize_k_cache_paged,
    dequantize_k_cache_paged_selective,
)


def sample(fn, repeats=15):
    for _ in range(4):
        value = fn()
    torch.cuda.synchronize()
    samples = []
    peak = 0
    for _ in range(repeats):
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        start = time.perf_counter()
        value = fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000)
        peak = max(peak, torch.cuda.max_memory_allocated() - before)
    return statistics.median(samples), peak, value


torch.manual_seed(424242)
rows = 131072
width = 2048
cache = torch.zeros((rows, 1, 656), device="cuda", dtype=torch.float8_e4m3fn)
page_table = torch.randperm(rows, device="cuda", dtype=torch.int32)
print("module", inspect.getfile(dequantize_k_cache_paged_selective))
print("device", torch.cuda.get_device_name())
full_ms, full_peak, full = sample(lambda: dequantize_k_cache_paged(cache, page_table))
print("full", f"ms={full_ms:.4f}", f"peak={full_peak}")
del full

for queries in (1, 4, 8, 16, 24, 32, 33, 64):
    random_topk = torch.randint(
        0, rows, (queries, width), device="cuda", dtype=torch.int32
    )
    ms, peak, result = sample(
        lambda topk=random_topk: dequantize_k_cache_paged_selective(
            cache, page_table, topk
        )
    )
    compact, remapped, used = result
    print(
        "random",
        f"queries={queries}",
        f"used={used}",
        f"rows={compact.shape[0]}",
        f"ms={ms:.4f}",
        f"ratio={ms / full_ms:.3f}",
        f"peak={peak}",
    )
    del random_topk, compact, remapped, result

shared = torch.arange(width, device="cuda", dtype=torch.int32).repeat(33, 1)
for label, kwargs in (("shared", {}), ("shared_forced_union", {"max_topk_ratio": 1.0})):
    ms, peak, result = sample(
        lambda kwargs=kwargs: dequantize_k_cache_paged_selective(
            cache, page_table, shared, **kwargs
        )
    )
    compact, remapped, used = result
    print(
        label,
        "queries=33",
        f"used={used}",
        f"rows={compact.shape[0]}",
        f"ms={ms:.4f}",
        f"ratio={ms / full_ms:.3f}",
        f"peak={peak}",
    )
