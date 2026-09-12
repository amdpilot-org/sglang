import inspect
import statistics
import time

import torch

from sglang.kernels.ops.attention.dsa.dequant_k_cache import (
    dequantize_k_cache_paged,
    dequantize_k_cache_paged_selective,
)


def measured(fn, repeats=21):
    for _ in range(6):
        value = fn()
    torch.cuda.synchronize()
    timings = []
    peak = 0
    for _ in range(repeats):
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        start = time.perf_counter_ns()
        value = fn()
        torch.cuda.synchronize()
        timings.append((time.perf_counter_ns() - start) / 1e6)
        peak = max(peak, torch.cuda.max_memory_allocated() - before)
    return statistics.median(timings), peak, value


torch.manual_seed(90210)
rows = 131072
topk_width = 2048
cache = torch.zeros((rows, 1, 656), device="cuda", dtype=torch.float8_e4m3fn)
logical_to_physical = torch.randperm(rows, device="cuda", dtype=torch.int32)

print("source", inspect.getfile(dequantize_k_cache_paged_selective))
print("device", torch.cuda.get_device_name(0))
full_ms, full_peak, full_result = measured(
    lambda: dequantize_k_cache_paged(cache, logical_to_physical)
)
print("full", f"ms={full_ms:.4f}", f"peak={full_peak}", f"rows={full_result.shape[0]}")
del full_result

for queries in (1, 2, 4, 8, 16, 24, 32):
    selected = torch.randint(
        0, rows, (queries, topk_width), device="cuda", dtype=torch.int32
    )
    latency, peak, outcome = measured(
        lambda selected=selected: dequantize_k_cache_paged_selective(
            cache, logical_to_physical, selected
        )
    )
    compact, remapped, used = outcome
    assert remapped.shape == selected.shape
    print(
        "random",
        f"queries={queries}",
        f"used={used}",
        f"rows={compact.shape[0]}",
        f"ms={latency:.4f}",
        f"vs_full={latency / full_ms:.3f}",
        f"peak={peak}",
    )
    del selected, compact, remapped, outcome

shared = torch.arange(topk_width, device="cuda", dtype=torch.int32).repeat(33, 1)
for label, extra in (
    ("shared_default", {}),
    ("shared_forced", {"max_topk_ratio": 1.0}),
):
    latency, peak, outcome = measured(
        lambda extra=extra: dequantize_k_cache_paged_selective(
            cache, logical_to_physical, shared, **extra
        )
    )
    compact, remapped, used = outcome
    assert remapped.shape == shared.shape
    print(
        label,
        f"used={used}",
        f"rows={compact.shape[0]}",
        f"ms={latency:.4f}",
        f"vs_full={latency / full_ms:.3f}",
        f"peak={peak}",
    )
