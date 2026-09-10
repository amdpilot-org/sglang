import pytest
import torch

from sglang.kernels.ops.quantization.fp8_kernel import is_fp8_fnuz
from sglang.kernels.ops.kvcache.fused_fp8_qkv_kv_cache import fused_fp8_qkv_kv_cache
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=40, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=40, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

FP8 = torch.float8_e4m3fnuz if is_fp8_fnuz() else torch.float8_e4m3fn


def _ref_quant(x_f32: torch.Tensor, inv_scale: float) -> torch.Tensor:
    fp8_max = torch.finfo(FP8).max
    y = (x_f32 * inv_scale).clamp(-fp8_max, fp8_max)
    return y.to(FP8)


def _bytes(t: torch.Tensor) -> torch.Tensor:
    return t.reshape(-1).view(torch.uint8)


def _encode_e4m3_fnuz_reference(values: torch.Tensor) -> torch.Tensor:
    """Independent E4M3FNUZ encoder: round-to-nearest-even and finite saturation."""
    bits = values.to(torch.float32).view(torch.int32)
    sign = (bits < 0).to(torch.uint8) << 7
    magnitude = (bits & 0x7FFFFFFF).to(torch.int64)
    exponent = (magnitude >> 23) & 0xFF
    mantissa = magnitude & 0x7FFFFF

    def round_rne(
        upper: torch.Tensor, remainder: torch.Tensor, half: int
    ) -> torch.Tensor:
        upper = upper.clone()
        round_up = remainder > half
        tie_odd = (remainder == half) & (upper & 1).bool()
        upper = upper + round_up.to(torch.int64) + tie_odd.to(torch.int64)
        return upper

    encoded = torch.zeros_like(sign)
    normal = exponent >= 120
    subnormal = (exponent >= 113) & ~normal

    sub_shift = exponent - 117
    sub_significand = (0x800000 + mantissa)[subnormal]
    sub_shifted = sub_significand >> (23 - sub_shift[subnormal])
    sub_remainder = sub_significand & ((1 << (23 - sub_shift[subnormal])) - 1)
    sub_half = 1 << (22 - sub_shift[subnormal])
    sub_mant = round_rne(sub_shifted, sub_remainder, sub_half)
    sub_code = sub_mant.clamp(max=7).to(torch.uint8)
    became_normal = sub_mant > 7
    sub_code = torch.where(became_normal, torch.full_like(sub_code, 0x08), sub_code)
    encoded[subnormal] = sub_code

    normal_exponent = (exponent[normal] - 127 + 8).to(torch.int64)
    normal_mant = round_rne(mantissa[normal] >> 20, mantissa[normal] & 0xFFFFF, 0x80000)
    normal_exponent = normal_exponent + (normal_mant > 7).to(torch.int64)
    normal_mant = torch.where(normal_mant > 7, torch.zeros_like(normal_mant), normal_mant)
    saturated = normal_exponent >= 15
    normal_code = ((normal_exponent << 3) + normal_mant).clamp(max=0x7F).to(torch.uint8)
    normal_code = torch.where(saturated, torch.full_like(normal_code, 0x7F), normal_code)
    encoded[normal] = normal_code

    return torch.where(encoded == 0, torch.zeros_like(encoded), sign | encoded)


def _decode_e4m3_fnuz_reference(encoded: torch.Tensor) -> torch.Tensor:
    sign = torch.where(encoded & 0x80 != 0, -1.0, 1.0).to(torch.float32)
    magnitude = encoded & 0x7F
    exponent = (magnitude >> 3).to(torch.float32)
    mantissa = (magnitude & 0x7).to(torch.float32)
    value = torch.where(
        exponent == 0,
        mantissa * 2.0**-10,
        (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 8),
    ).to(torch.float32)
    return sign * value


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


@pytest.mark.skipif(not is_fp8_fnuz(), reason="E4M3FNUZ is the native MI300X format")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fused_fp8_qkv_kv_cache_fnuz_fidelity(dtype):
    torch.manual_seed(0)
    device = "cuda"
    page_size = 3
    num_pages = 5
    num_heads = 2
    head_dim = 65
    kv_dim = num_heads * head_dim
    num_tokens = 8
    k_stride = kv_dim + 16
    v_stride = kv_dim + 32

    k_storage = torch.randn((num_tokens, k_stride), dtype=dtype, device=device)
    v_storage = torch.randn((num_tokens, v_stride), dtype=dtype, device=device)
    k = torch.as_strided(k_storage, (num_tokens, kv_dim), (k_stride, 1))
    v = torch.as_strided(v_storage, (num_tokens, kv_dim), (v_stride, 1))
    k[[0, 3]] = 0
    v[[2, 5]] = 0
    k[1, 0] = 10000.0
    k[4, 1] = -10000.0
    v[6, 2] = 10000.0
    v[7, 3] = -10000.0

    cache_shape = (num_pages, page_size, num_heads, head_dim)
    k_cache_bytes = torch.randint(
        0, 256, (num_pages * page_size * kv_dim,), dtype=torch.uint8, device=device
    )
    v_cache_bytes = torch.randint(
        0, 256, (num_pages * page_size * kv_dim,), dtype=torch.uint8, device=device
    )
    k_cache = k_cache_bytes.view(FP8).view(cache_shape)
    v_cache = v_cache_bytes.view(FP8).view(cache_shape)
    k_cache_before = k_cache.view(torch.uint8).clone()
    v_cache_before = v_cache.view(torch.uint8).clone()

    cache_loc = torch.tensor(
        [0, 2, 3, 4, 5, 8, 9, 12], dtype=torch.int64, device=device
    )
    k_scale = torch.tensor([1.7], dtype=torch.float32, device=device)
    v_scale = torch.tensor([0.9], dtype=torch.float32, device=device)

    assert fused_fp8_qkv_kv_cache(
        None, k, v, k_cache, v_cache, cache_loc, k_scale, v_scale
    ) is None

    k_expected = _encode_e4m3_fnuz_reference(k.float() * (1.0 / k_scale.item()))
    v_expected = _encode_e4m3_fnuz_reference(v.float() * (1.0 / v_scale.item()))
    k_actual = k_cache.view(torch.uint8).reshape(-1, kv_dim)[cache_loc]
    v_actual = v_cache.view(torch.uint8).reshape(-1, kv_dim)[cache_loc]
    assert torch.equal(k_actual, k_expected)
    assert torch.equal(v_actual, v_expected)

    assert torch.equal(k_actual[[0, 3]], torch.zeros_like(k_actual[[0, 3]]))
    assert torch.equal(v_actual[[2, 5]], torch.zeros_like(v_actual[[2, 5]]))

    unused = torch.ones(num_pages * page_size, dtype=torch.bool, device=device)
    unused[cache_loc] = False
    assert torch.equal(
        k_cache.view(torch.uint8).reshape(-1, kv_dim)[unused],
        k_cache_before.reshape(-1, kv_dim)[unused],
    )
    assert torch.equal(
        v_cache.view(torch.uint8).reshape(-1, kv_dim)[unused],
        v_cache_before.reshape(-1, kv_dim)[unused],
    )

    torch.testing.assert_close(
        k_cache.view(-1, kv_dim)[cache_loc].float(),
        _decode_e4m3_fnuz_reference(k_expected),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        v_cache.view(-1, kv_dim)[cache_loc].float(),
        _decode_e4m3_fnuz_reference(v_expected),
        rtol=0.0,
        atol=0.0,
    )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
