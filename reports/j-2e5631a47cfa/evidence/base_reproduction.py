import inspect
import torch

from sglang.kernels.ops.attention.dsa import dequant_k_cache


print(f"module={inspect.getfile(dequant_k_cache)}")
print(f"device={torch.cuda.get_device_name(0)}")

rows = 131072
cache = torch.zeros((rows, 1, 656), device="cuda", dtype=torch.float8_e4m3fn)
page_table = torch.arange(rows, device="cuda", dtype=torch.int32)
out = dequant_k_cache.dequantize_k_cache_paged(cache, page_table)
torch.cuda.synchronize()
print(f"logical_rows={rows} output_rows={out.shape[0]} output_bytes={out.numel() * out.element_size()}")
print(f"selective_api_present={hasattr(dequant_k_cache, 'dequantize_k_cache_paged_selective')}")
