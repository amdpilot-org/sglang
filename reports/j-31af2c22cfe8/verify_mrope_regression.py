"""Focused numerical check for the Qwen3.5 fused mRoPE regression."""

import torch

import sglang.kernels.ops.attention.fused_qk_rmsnorm_rope_gate as fused_module
from sglang.srt.layers.rotary_embedding.mrope import MRotaryEmbedding


def rmsnorm(x, weight, eps):
    value = x.float()
    value = value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + eps)
    return (value * (1 + weight.float())).to(x.dtype)


def main():
    torch.manual_seed(35345)
    device = "cuda"
    dtype = torch.bfloat16
    tokens, q_heads, kv_heads, head_dim = 17, 4, 2, 128
    section = [11, 11, 10]
    rotary_dim = 2 * sum(section)
    eps = 1e-6

    # HIP exposes its device through torch.cuda. Prevent this CUDA-only optional
    # instruction from obscuring the mRoPE numerical check on the assigned GPU.
    if torch.version.hip is not None:
        fused_module._ENABLE_PDL = False

    rope = MRotaryEmbedding(
        head_size=head_dim,
        rotary_dim=rotary_dim,
        max_position_embeddings=512,
        base=10_000_000,
        is_neox_style=True,
        dtype=dtype,
        mrope_section=section,
        mrope_interleaved=True,
    ).to(device)

    q_gate = torch.randn(tokens, q_heads * 2 * head_dim, device=device, dtype=dtype)
    key = torch.randn(tokens, kv_heads * head_dim, device=device, dtype=dtype)
    q_weight = torch.randn(head_dim, device=device, dtype=dtype)
    k_weight = torch.randn(head_dim, device=device, dtype=dtype)
    packed_q = q_gate.view(tokens, q_heads, 2 * head_dim)[..., :head_dim]
    normalized_q = rmsnorm(packed_q, q_weight, eps).reshape(tokens, -1)
    normalized_k = rmsnorm(key.view(tokens, kv_heads, head_dim), k_weight, eps).reshape(
        tokens, -1
    )

    temporal = torch.arange(tokens, device=device) % 7
    distinct = torch.stack(
        [
            temporal,
            torch.arange(tokens, device=device) % 5 + 3,
            torch.arange(tokens, device=device) % 3 + 11,
        ]
    )
    equal = temporal.repeat(3, 1)

    def fused(positions, axis_map):
        return fused_module.fused_qk_gemma_rmsnorm_rope_gate(
            q_gate,
            key,
            q_weight,
            k_weight,
            rope.cos_sin_cache,
            positions,
            eps,
            q_heads,
            kv_heads,
            head_dim,
            rotary_dim,
            mrope_axis_map=axis_map,
        )[:2]

    reference_q, reference_k = rope.forward_native(distinct, normalized_q, normalized_k)
    fixed_q, fixed_k = fused(distinct, rope.axis_map)
    old_q, old_k = fused(distinct[0], None)
    equal_q, equal_k = fused(equal, rope.axis_map)
    flat_q, flat_k = fused(temporal, None)

    fixed_error = max(
        (fixed_q - reference_q).abs().float().max().item(),
        (fixed_k - reference_k).abs().float().max().item(),
    )
    old_error = max(
        (old_q - reference_q).abs().float().max().item(),
        (old_k - reference_k).abs().float().max().item(),
    )
    equal_error = max(
        (equal_q - flat_q).abs().float().max().item(),
        (equal_k - flat_k).abs().float().max().item(),
    )
    print(f"fixed_distinct_axis_max_abs_error={fixed_error:.8f}")
    print(f"old_temporal_only_max_abs_error={old_error:.8f}")
    print(f"equal_axis_vs_1d_max_abs_error={equal_error:.8f}")
    torch.testing.assert_close(
        fixed_q.float(), reference_q.float(), atol=2e-2, rtol=2e-2
    )
    torch.testing.assert_close(
        fixed_k.float(), reference_k.float(), atol=2e-2, rtol=2e-2
    )
    torch.testing.assert_close(equal_q.float(), flat_q.float(), atol=0, rtol=0)
    torch.testing.assert_close(equal_k.float(), flat_k.float(), atol=0, rtol=0)
    assert old_error > 0.1, old_error


if __name__ == "__main__":
    main()
