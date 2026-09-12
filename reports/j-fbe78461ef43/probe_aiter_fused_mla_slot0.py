import torch

from aiter.ops.triton.fused_kv_cache import fused_qk_rope_cat_and_cache_mla


def main():
    dtype = torch.bfloat16
    q_nope = torch.zeros((1, 16, 512), dtype=dtype, device="cuda")
    q_pe = torch.zeros((1, 16, 64), dtype=dtype, device="cuda")
    k_nope = torch.full((2, 1, 512), 2, dtype=dtype, device="cuda")
    k_pe = torch.full((2, 1, 64), 3, dtype=dtype, device="cuda")
    k_nope[0].fill_(torch.nan)
    k_pe[0].fill_(torch.nan)
    kv_cache = torch.full((4, 1, 576), 17, dtype=dtype, device="cuda")
    slot_mapping = torch.tensor([0, 1], dtype=torch.int64, device="cuda")
    positions = torch.zeros(1, dtype=torch.int64, device="cuda")
    cos = torch.ones((1, 64), dtype=dtype, device="cuda")
    sin = torch.zeros_like(cos)
    k_scale = torch.ones((), dtype=torch.float32, device="cuda")
    slot0_before = kv_cache[0].clone()

    fused_qk_rope_cat_and_cache_mla(
        q_nope,
        q_pe,
        k_nope,
        k_pe,
        kv_cache,
        slot_mapping,
        positions,
        cos,
        sin,
        k_scale,
        is_neox=True,
    )
    torch.cuda.synchronize()

    print(f"device={torch.cuda.get_device_name(0)}")
    print(f"arch={torch.cuda.get_device_properties(0).gcnArchName}")
    print(f"slot0_unchanged={torch.equal(kv_cache[0], slot0_before)}")
    print(f"slot0_nan_count={torch.isnan(kv_cache[0]).sum().item()}")
    print(f"slot1_finite={torch.isfinite(kv_cache[1]).all().item()}")
    assert not torch.equal(kv_cache[0], slot0_before)
    assert torch.isnan(kv_cache[0]).all()
    assert torch.isfinite(kv_cache[1]).all()


if __name__ == "__main__":
    main()
