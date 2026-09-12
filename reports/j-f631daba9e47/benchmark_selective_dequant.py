import time

import torch

from sglang.kernels.ops.attention.dsa.dequant_k_cache import (
    dequantize_k_cache_paged,
    dequantize_k_cache_paged_selective,
)


def measure(fn, warmup=5, iterations=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000 / iterations


torch.manual_seed(19)
device = torch.device("cuda")
num_tokens = 262144
union_tokens = 4096
query_tokens = 8
topk_width = 2048

# The dequant kernel consumes packed bytes; random bytes are sufficient for a
# performance-only run because numerical behavior is covered separately.
cache = torch.randint(
    0, 255, (num_tokens, 1, 656), device=device, dtype=torch.uint8
).view(torch.float8_e4m3fn)
page_table = torch.randperm(num_tokens, device=device, dtype=torch.int32)
topk = torch.randint(
    0, union_tokens, (query_tokens, topk_width), device=device, dtype=torch.int32
)

full_ms = measure(lambda: dequantize_k_cache_paged(cache, page_table))

selected_rows = []


def selective():
    kv, _, used = dequantize_k_cache_paged_selective(cache, page_table, topk)
    selected_rows.append((kv.shape[0], used))


selective_ms = measure(selective)
rows, used = selected_rows[-1]
print(f"device={torch.cuda.get_device_name(0)}")
print(f"logical_rows={num_tokens} compact_rows={rows} selective_used={used}")
print(f"full_bf16_bytes={num_tokens * 576 * 2} compact_bf16_bytes={rows * 576 * 2}")
print(
    f"full_ms={full_ms:.4f} selective_ms={selective_ms:.4f} speedup={full_ms / selective_ms:.3f}x"
)
