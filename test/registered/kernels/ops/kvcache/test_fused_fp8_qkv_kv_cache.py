import pytest
import torch

from sglang.kernels.ops.kvcache.fused_fp8_qkv_kv_cache import fused_fp8_qkv_kv_cache
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=40, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=40, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

FP8 = torch.float8_e4m3fnuz if torch.version.hip else torch.float8_e4m3fn


def _ref_quant(x_f32: torch.Tensor, inv_scale: float) -> torch.Tensor:
    fp8_info = torch.finfo(FP8)
    y = (x_f32 * inv_scale).clamp(fp8_info.min, fp8_info.max)
    return y.to(FP8)


def _bytes(t: torch.Tensor) -> torch.Tensor:
    return t.reshape(-1).view(torch.uint8)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "hq,hkv,head_dim", [(8, 1, 128), (8, 8, 128), (4, 2, 64), (64, 2, 128)]
)
@pytest.mark.parametrize(
    "num_tokens", [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
)
@pytest.mark.parametrize("scale", [None, 0.5, 2.0])
@pytest.mark.parametrize("fused_qkv", [False, True])
@pytest.mark.parametrize("quantize_q", [True, False])
def test_fused_fp8_qkv_kv_cache(
    dtype, hq, hkv, head_dim, num_tokens, scale, fused_qkv, quantize_q
):
    idx_dtype = torch.int64
    torch.manual_seed(0)
    device = "cuda"
    q_dim = hq * head_dim
    kv_dim = hkv * head_dim
    total_slots = num_tokens + 4

    if fused_qkv:
        qkv = torch.randn(num_tokens, q_dim + 2 * kv_dim, dtype=dtype, device=device)
        q = qkv[:, :q_dim]
        k = qkv[:, q_dim : q_dim + kv_dim].view(num_tokens, hkv, head_dim)
        v = qkv[:, q_dim + kv_dim :].view(num_tokens, hkv, head_dim)
        if num_tokens > 1:
            assert not q.is_contiguous()
    else:
        q = torch.randn(num_tokens, q_dim, dtype=dtype, device=device)
        k = torch.randn(num_tokens, hkv, head_dim, dtype=dtype, device=device)
        v = torch.randn(num_tokens, hkv, head_dim, dtype=dtype, device=device)
    k_cache = torch.zeros(total_slots, hkv, head_dim, dtype=FP8, device=device)
    v_cache = torch.zeros(total_slots, hkv, head_dim, dtype=FP8, device=device)

    cache_loc = torch.randperm(total_slots, device=device)[:num_tokens].to(idx_dtype)

    if scale is None:
        k_scale = v_scale = None
        inv_k = inv_v = 1.0
    else:
        k_scale = torch.tensor(scale, dtype=torch.float32, device=device)
        v_scale = torch.tensor(scale * 1.5, dtype=torch.float32, device=device)
        inv_k = 1.0 / float(k_scale)
        inv_v = 1.0 / float(v_scale)

    q_out = fused_fp8_qkv_kv_cache(
        q if quantize_q else None, k, v, k_cache, v_cache, cache_loc, k_scale, v_scale
    )

    if quantize_q:
        q_ref = q.to(FP8)
        torch.testing.assert_close(_bytes(q_out), _bytes(q_ref), rtol=0, atol=0)
    else:
        assert q_out is None

    k_ref = _ref_quant(k.reshape(num_tokens, kv_dim).float(), inv_k)
    v_ref = _ref_quant(v.reshape(num_tokens, kv_dim).float(), inv_v)
    loc = cache_loc.long()
    torch.testing.assert_close(
        _bytes(k_cache.reshape(total_slots, kv_dim)[loc]), _bytes(k_ref), rtol=0, atol=0
    )
    torch.testing.assert_close(
        _bytes(v_cache.reshape(total_slots, kv_dim)[loc]), _bytes(v_ref), rtol=0, atol=0
    )


def test_fused_fp8_qkv_scale_axes_are_independent():
    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.bfloat16
    num_tokens, hq, hkv, head_dim = 11, 5, 3, 17
    total_slots = num_tokens + 6
    q_dim = hq * head_dim
    kv_dim = hkv * head_dim

    q = torch.randn(num_tokens, hq, head_dim, dtype=dtype, device=device)
    k = torch.randn(num_tokens, hkv, head_dim, dtype=dtype, device=device)
    v = torch.randn(num_tokens, hkv, head_dim, dtype=dtype, device=device)
    sentinel = torch.tensor(0x55, dtype=torch.uint8, device=device)
    k_cache = sentinel.repeat(total_slots * kv_dim).view(total_slots, kv_dim)
    v_cache = sentinel.repeat(total_slots * kv_dim).view(total_slots, kv_dim)
    cache_loc = torch.randperm(total_slots, device=device)[:num_tokens].to(torch.int64)

    q_scale = torch.tensor([0.03125], dtype=torch.float32, device=device)
    k_scale = torch.tensor([[0.0625]], dtype=torch.float32, device=device)
    v_scale = torch.tensor([0.125], dtype=torch.float32, device=device)
    q_out = fused_fp8_qkv_kv_cache(
        q,
        k,
        v,
        k_cache,
        v_cache,
        cache_loc,
        k_scale=k_scale,
        v_scale=v_scale,
        q_scale=q_scale,
    )

    fp8_info = torch.finfo(FP8)
    q_ref = q.reshape(num_tokens, q_dim).float() / q_scale
    k_ref = k.reshape(num_tokens, kv_dim).float() / k_scale
    v_ref = v.reshape(num_tokens, kv_dim).float() / v_scale
    q_ref = q_ref.clamp(fp8_info.min, fp8_info.max).to(FP8)
    k_ref = k_ref.clamp(fp8_info.min, fp8_info.max).to(FP8)
    v_ref = v_ref.clamp(fp8_info.min, fp8_info.max).to(FP8)

    torch.testing.assert_close(_bytes(q_out), _bytes(q_ref), rtol=0, atol=0)
    torch.testing.assert_close(q_out.float() * q_scale, q_ref.float() * q_scale)
    loc = cache_loc.long()
    torch.testing.assert_close(
        _bytes(k_cache.reshape(total_slots, kv_dim)[loc]), _bytes(k_ref), rtol=0, atol=0
    )
    torch.testing.assert_close(
        _bytes(v_cache.reshape(total_slots, kv_dim)[loc]), _bytes(v_ref), rtol=0, atol=0
    )
    torch.testing.assert_close(
        k_cache.reshape(total_slots, kv_dim)[loc].view(FP8).float() * k_scale,
        k_ref.float() * k_scale,
    )
    torch.testing.assert_close(
        v_cache.reshape(total_slots, kv_dim)[loc].view(FP8).float() * v_scale,
        v_ref.float() * v_scale,
    )

    used = torch.zeros(total_slots, dtype=torch.bool, device=device)
    used[loc] = True
    unused_count = int((~used).sum())
    torch.testing.assert_close(
        _bytes(k_cache[~used]),
        sentinel.repeat(unused_count * kv_dim),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        _bytes(v_cache[~used]),
        sentinel.repeat(unused_count * kv_dim),
        rtol=0,
        atol=0,
    )
    assert torch.isfinite(sentinel.view(FP8).float()).all()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
