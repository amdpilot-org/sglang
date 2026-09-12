"""Numerically compare the fixed Qwen3.5 fused mRoPE path with the legacy behavior.

This is a kernel-level transport/geometry regression, not a model-quality test.
It requires one GPU and uses synthetic deterministic tensors.
"""

import json

import torch

import sglang.kernels.ops.attention.fused_qk_rmsnorm_rope_gate as fused_module
from sglang.srt.layers.rotary_embedding.mrope import MRotaryEmbedding


def gemma_rmsnorm(x, weight, eps):
    value = x.float()
    value = value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + eps)
    return (value * (1.0 + weight.float())).to(x.dtype)


def run_fused(q_gate, key, q_weight, k_weight, rope, positions):
    return fused_module.fused_qk_gemma_rmsnorm_rope_gate(
        q_gate,
        key,
        q_weight,
        k_weight,
        rope.cos_sin_cache,
        positions,
        1e-6,
        8,
        2,
        128,
        rope.rotary_dim,
        mrope_axis_map=rope.axis_map if positions.ndim == 2 else None,
    )


def main():
    torch.manual_seed(35772)
    device, dtype, tokens = "cuda", torch.bfloat16, 17
    q_gate = torch.randn(tokens, 8 * 2 * 128, device=device, dtype=dtype)
    key = torch.randn(tokens, 2 * 128, device=device, dtype=dtype)
    q_weight = torch.randn(128, device=device, dtype=dtype)
    k_weight = torch.randn(128, device=device, dtype=dtype)
    rope = MRotaryEmbedding(
        head_size=128,
        rotary_dim=64,
        max_position_embeddings=512,
        base=10000,
        is_neox_style=True,
        dtype=dtype,
        mrope_section=[11, 11, 10],
        mrope_interleaved=True,
    ).to(device)

    storage = torch.zeros(3, tokens * 4, dtype=torch.int64, device=device)
    storage[:, :tokens] = torch.stack(
        (
            torch.arange(tokens, device=device) % 7,
            torch.arange(tokens, device=device) % 5 + 3,
            torch.arange(tokens, device=device) % 3 + 11,
        )
    )
    positions = storage[:, :tokens]

    packed_q = q_gate.view(tokens, 8, 256)[..., :128]
    reference_q, _ = rope.forward_native(
        positions,
        gemma_rmsnorm(packed_q, q_weight, 1e-6).reshape(tokens, -1),
        gemma_rmsnorm(key.view(tokens, 2, 128), k_weight, 1e-6).reshape(tokens, -1),
    )

    # PDL is CUDA-only. The production ROCm Qwen3.5 path does not dispatch here;
    # disable it so gfx950 can still validate the kernel's mRoPE arithmetic.
    fused_module._ENABLE_PDL = False
    fixed_q, _, _ = run_fused(q_gate, key, q_weight, k_weight, rope, positions)
    legacy_q, _, _ = run_fused(q_gate, key, q_weight, k_weight, rope, positions[0])

    equal_positions = positions[0].repeat(3, 1)
    equal_fixed_q, _, _ = run_fused(
        q_gate, key, q_weight, k_weight, rope, equal_positions
    )
    equal_flat_q, _, _ = run_fused(
        q_gate, key, q_weight, k_weight, rope, equal_positions[0]
    )

    result = {
        "device": torch.cuda.get_device_name(0),
        "architecture": torch.cuda.get_device_properties(0).gcnArchName,
        "positions_shape": list(positions.shape),
        "positions_stride": list(positions.stride()),
        "fixed_vs_native_max_abs": float((fixed_q - reference_q).abs().max()),
        "legacy_temporal_only_vs_native_max_abs": float(
            (legacy_q - reference_q).abs().max()
        ),
        "equal_axis_2d_vs_1d_max_abs": float(
            (equal_fixed_q - equal_flat_q).abs().max()
        ),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
