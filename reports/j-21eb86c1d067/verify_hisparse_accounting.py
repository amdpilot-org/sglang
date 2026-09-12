from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.mem_cache.memory_pool import GB


def owned_tensors(pool):
    tensors = []
    if pool._unified_kv:
        tensors.extend(pool.unified_kv_pool.kv_buffer)
    else:
        tensors.extend(pool.swa_kv_pool.kv_buffer)
        for child in pool.kv_pools.values():
            if child is not None:
                tensors.extend(child.kv_buffer)
                if hasattr(child, "data_ptrs"):
                    tensors.append(child.data_ptrs)
    for child in pool.index_pools.values():
        tensors.extend(child.contiguous_page_row_buffers())
    for children in (pool.compress_state_pools, pool.indexer_compress_state_pools):
        tensors.extend(
            child.kv_score_buffer.kv_score for child in children if child is not None
        )
    if pool.online_c128_mtp_pending_seq_lens is not None:
        tensors.append(pool.online_c128_mtp_pending_seq_lens)
    return tensors


def make_pool(*, unified=False, hisparse=False, fp4=False):
    exec_ctx = SimpleNamespace(
        kernel=SimpleNamespace(
            enable_deepseek_v4_fp4_indexer=fp4,
            dsv4_attn_backend="flashmla",
        )
    )
    spec_ctx = SimpleNamespace(speculative_algorithm=None)
    with (
        patch(
            "sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate.is_unified_kv_triton",
            return_value=unified,
        ),
        patch(
            "sglang.srt.mem_cache.deepseek_v4_memory_pool.get_exec",
            return_value=exec_ctx,
        ),
        patch(
            "sglang.srt.mem_cache.deepseek_v4_memory_pool.get_spec",
            return_value=spec_ctx,
        ),
    ):
        return DeepSeekV4TokenToKVPool(
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
            enable_hisparse=hisparse,
        )


for label, kwargs in (
    ("non_unified_fp8", {}),
    ("non_unified_fp4", {"fp4": True}),
    ("hisparse", {"hisparse": True}),
    ("unified", {"unified": True}),
):
    torch.cuda.empty_cache()
    pool = make_pool(**kwargs)
    expected = sum(t.nbytes for t in owned_tensors(pool))
    reported = round(pool.mem_usage * GB)
    print(f"{label}: expected={expected} reported={reported} delta={expected-reported}")
    assert reported == expected

print(f"device={torch.cuda.get_device_name(0)}")
