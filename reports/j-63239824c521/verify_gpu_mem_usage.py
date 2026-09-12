from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
    GB,
    DeepSeekV4TokenToKVPool,
)


def tensor_bytes(pool: DeepSeekV4TokenToKVPool) -> int:
    tensors = []
    tensors.extend(pool.swa_kv_pool.kv_buffer)
    for child in pool.kv_pools.values():
        if child is not None:
            tensors.extend(child.kv_buffer)
    for child in pool.index_pools.values():
        tensors.extend(child.contiguous_page_row_buffers())
    for children in (pool.compress_state_pools, pool.indexer_compress_state_pools):
        tensors.extend(
            child.kv_score_buffer.kv_score
            for child in children
            if child is not None
        )
    return sum(tensor.nbytes for tensor in tensors)


exec_ctx = SimpleNamespace(
    kernel=SimpleNamespace(
        enable_deepseek_v4_fp4_indexer=False,
        dsv4_attn_backend="flashmla",
    )
)
spec_ctx = SimpleNamespace(speculative_algorithm=None)
with (
    patch(
        "sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate.is_unified_kv_triton",
        return_value=False,
    ),
    patch("sglang.srt.mem_cache.deepseek_v4_memory_pool.get_exec", return_value=exec_ctx),
    patch("sglang.srt.mem_cache.deepseek_v4_memory_pool.get_spec", return_value=spec_ctx),
):
    pool = DeepSeekV4TokenToKVPool(
        max_num_reqs=2,
        swa_size=256,
        c4_size=64,
        c128_size=2,
        c4_state_pool_size=32,
        c128_state_pool_size=256,
        page_size=128,
        swa_page_size=128,
        dtype=torch.float8_e4m3fn,
        c4_state_dtype=torch.bfloat16,
        c128_state_dtype=torch.float32,
        qk_nope_head_dim=448,
        qk_rope_head_dim=64,
        indexer_head_dim=128,
        layer_num=3,
        device="cuda",
        enable_memory_saver=False,
        compression_ratios=[0, 4, 128],
    )

expected_bytes = tensor_bytes(pool)
measured_bytes = round(pool.mem_usage * GB)
assert measured_bytes == expected_bytes, (measured_bytes, expected_bytes)
assert pool.mem_usage > 0
print(f"device={torch.cuda.get_device_name(0)}")
print(f"expected_tensor_bytes={expected_bytes}")
print(f"reported_mem_usage_gb={pool.mem_usage:.12f}")
