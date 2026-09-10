import pytest
import torch

from sglang.kernels.ops.kvcache import reshape_and_cache_flash
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=5, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=5, stage="jit-kernel-unit", runner_config="amd")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="reshape_and_cache_flash requires a GPU"
)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("scale_mode", [None, "both", "key_only", "value_only"])
def test_reshape_and_cache_flash_public_dispatch(dtype, scale_mode):
    device = "cuda"
    num_tokens, num_heads, head_size = 7, 3, 17
    page_size, num_pages = 5, 5
    cache_shape = (num_pages, page_size, num_heads, head_size)
    slots = torch.tensor([0, 4, 5, 9, 14, 19, 24], device=device, dtype=torch.int64)

    key = torch.arange(num_tokens * num_heads * head_size, device=device)
    key = key.reshape(num_tokens, num_heads, head_size).to(dtype)
    value = 10_000 + torch.arange(num_tokens * num_heads * head_size, device=device)
    value = value.reshape(num_tokens, num_heads, head_size).to(dtype)

    sentinel = torch.full(cache_shape, -12345, device=device, dtype=dtype)
    key_cache = sentinel.clone()
    value_cache = sentinel.clone()

    key_scale = torch.tensor([2.0], device=device, dtype=torch.float32)
    value_scale = torch.tensor([4.0], device=device, dtype=torch.float32)
    reshape_and_cache_flash(
        key,
        value,
        key_cache,
        value_cache,
        slots,
        k_scale=key_scale if scale_mode in {"both", "key_only"} else None,
        v_scale=value_scale if scale_mode in {"both", "value_only"} else None,
    )

    expected_key = sentinel.clone()
    expected_value = sentinel.clone()
    used = torch.zeros(cache_shape, device=device, dtype=torch.bool)
    for token_idx, slot in enumerate(slots.tolist()):
        page_idx, page_offset = divmod(slot, page_size)
        expected_key[page_idx, page_offset] = key[token_idx]
        expected_value[page_idx, page_offset] = value[token_idx]
        used[page_idx, page_offset] = True
        if scale_mode in {"both", "key_only"}:
            expected_key[page_idx, page_offset] = key[token_idx] / key_scale
        if scale_mode in {"both", "value_only"}:
            expected_value[page_idx, page_offset] = value[token_idx] / value_scale

    assert torch.equal(key_cache, expected_key)
    assert torch.equal(value_cache, expected_value)
    assert torch.equal(key_cache[~used], sentinel[~used])
    assert torch.equal(value_cache[~used], sentinel[~used])
