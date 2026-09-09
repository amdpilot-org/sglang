import argparse

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

try:
    from sglang.kernels.ops.diffusion.common.numerics import round_bf16_to_fp32
except ImportError:
    from sglang.kernels.ops.diffusion.triton.numerics import round_bf16_to_fp32

try:
    from sglang.kernels.kda_kernels.layernorm_modulate_triton import (
        can_use_fused_layernorm_modulate,
        can_use_fused_qk_head_layernorm,
        fused_layernorm_modulate_raw,
        fused_qk_head_layernorm,
    )
except ImportError:
    try:
        from sglang.kernels.ops.diffusion.norm.layernorm_modulate_triton import (
            can_use_fused_layernorm_modulate,
            can_use_fused_qk_head_layernorm,
            fused_layernorm_modulate_raw,
            fused_qk_head_layernorm,
        )
    except ImportError:
        from sglang.kernels.ops.diffusion.triton.layernorm_modulate import (
            can_use_fused_layernorm_modulate,
            can_use_fused_qk_head_layernorm,
            fused_layernorm_modulate_raw,
            fused_qk_head_layernorm,
        )


def make_inputs(seed: int = 34351):
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    x = torch.randn(2, 3, 256, device="cuda", dtype=torch.bfloat16, generator=generator)
    scale = torch.randn(2, 256, device="cuda", dtype=torch.bfloat16, generator=generator)
    shift = torch.randn(2, 256, device="cuda", dtype=torch.bfloat16, generator=generator)
    q = torch.randn(2, 3, 8, 64, device="cuda", dtype=torch.bfloat16, generator=generator)
    k = torch.randn_like(q)
    return x, scale, shift, q, k


def direct_ln() -> None:
    x, scale, shift, _, _ = make_inputs()
    out = fused_layernorm_modulate_raw(x, scale, shift, 1e-5)
    torch.cuda.synchronize()
    print(out[:1, :1, :8].cpu().tolist())


def direct_qk() -> None:
    _, _, _, q, k = make_inputs()
    q_out, k_out = fused_qk_head_layernorm(q, k, 1e-5)
    torch.cuda.synchronize()
    print(q_out[:1, :1, :1, :8].cpu().tolist())
    print(k_out[:1, :1, :1, :8].cpu().tolist())


def explicit_ln_modulate(x, scale, shift, eps):
    x_f32 = x.float()
    mean = x_f32.mean(dim=-1, keepdim=True)
    variance = x_f32.var(dim=-1, unbiased=False, keepdim=True)
    normalized = ((x_f32 - mean) * torch.rsqrt(variance + eps)).to(torch.bfloat16)
    scale_term = (1.0 + scale.float()).to(torch.bfloat16)
    return normalized * scale_term.unsqueeze(1) + shift.unsqueeze(1)


def explicit_qk_ln(q, k, eps):
    def normalize(tensor):
        tensor_f32 = tensor.float()
        mean = tensor_f32.mean(dim=-1, keepdim=True)
        variance = tensor_f32.var(dim=-1, unbiased=False, keepdim=True)
        return ((tensor_f32 - mean) * torch.rsqrt(variance + eps)).to(torch.bfloat16)

    return normalize(q), normalize(k)


def fallback_ln_modulate(x, scale, shift, eps):
    normalized = F.layer_norm(x, (x.shape[-1],), None, None, eps)
    return normalized * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def report_tensor(label, tensor):
    print(f"{label}_head={tensor.flatten()[:8].detach().cpu().tolist()}")
    print(f"{label}_finite={bool(torch.isfinite(tensor).all().item())}")


def semantics() -> None:
    x, scale, shift, q, k = make_inputs()
    eps = 1e-5
    print(f"can_use_fused_layernorm_modulate={can_use_fused_layernorm_modulate(x, scale, shift)}")
    print(f"can_use_fused_qk_head_layernorm={can_use_fused_qk_head_layernorm(q, k)}")

    actual = fallback_ln_modulate(x, scale, shift, eps)
    torch_reference = fallback_ln_modulate(x, scale, shift, eps)
    explicit_reference = explicit_ln_modulate(x, scale, shift, eps)
    diff = (actual.float() - explicit_reference.float()).abs()
    print(f"ln_modulate_bitwise_equal_to_torch_reference={torch.equal(actual, torch_reference)}")
    print(f"ln_modulate_bitwise_equal_to_explicit_reference={torch.equal(actual, explicit_reference)}")
    print(f"ln_modulate_allclose_2e-2_2e-2={torch.allclose(actual.float(), explicit_reference.float(), rtol=2e-2, atol=2e-2)}")
    print(f"ln_modulate_max_abs_diff={diff.max().item():.9g}")
    report_tensor("ln_modulate_actual", actual)

    x_nan = x.clone()
    x_nan[0, 0, :] = float("nan")
    actual_nan = fallback_ln_modulate(x_nan, scale, shift, eps)
    explicit_nan = explicit_ln_modulate(x_nan, scale, shift, eps)
    print(f"ln_modulate_nan_mask_equal={torch.equal(torch.isnan(actual_nan), torch.isnan(explicit_nan))}")
    print(f"ln_modulate_nan_count={int(torch.isnan(actual_nan).sum().item())}")

    zero_scale = torch.zeros_like(scale)
    zero_scale[0, 0] = -1.0
    neg_zero_shift = torch.zeros_like(shift)
    neg_zero_shift[0, 0] = -0.0
    actual_zero = fallback_ln_modulate(x, zero_scale, neg_zero_shift, eps)
    explicit_zero = explicit_ln_modulate(x, zero_scale, neg_zero_shift, eps)
    print(f"ln_modulate_signed_zero_mask_equal={torch.equal(torch.signbit(actual_zero), torch.signbit(explicit_zero))}")
    print(f"ln_modulate_negative_zero_count={int(torch.signbit(actual_zero).sum().item())}")
    print(f"ln_modulate_zero_row_signbits={torch.signbit(actual_zero[0, 0, :16]).detach().cpu().tolist()}")

    actual_q, actual_k = F.layer_norm(q, (q.shape[-1],), None, None, eps), F.layer_norm(k, (k.shape[-1],), None, None, eps)
    explicit_q, explicit_k = explicit_qk_ln(q, k, eps)
    print(f"qk_ln_bitwise_equal_to_explicit_reference={torch.equal(actual_q, explicit_q) and torch.equal(actual_k, explicit_k)}")
    print(f"qk_ln_allclose_2e-2_2e-2={torch.allclose(actual_q.float(), explicit_q.float(), rtol=2e-2, atol=2e-2) and torch.allclose(actual_k.float(), explicit_k.float(), rtol=2e-2, atol=2e-2)}")
    print(f"qk_ln_max_abs_diff={max((actual_q.float()-explicit_q.float()).abs().max().item(), (actual_k.float()-explicit_k.float()).abs().max().item()):.9g}")
    report_tensor("qk_ln_actual", actual_q)

    q_nan = q.clone()
    q_nan[0, 0, 0, :] = float("nan")
    actual_q_nan = F.layer_norm(q_nan, (q_nan.shape[-1],), None, None, eps)
    explicit_q_nan, _ = explicit_qk_ln(q_nan, k, eps)
    print(f"qk_ln_nan_mask_equal={torch.equal(torch.isnan(actual_q_nan), torch.isnan(explicit_q_nan))}")
    print(f"qk_ln_nan_count={int(torch.isnan(actual_q_nan).sum().item())}")


@triton.jit
def round_bf16_kernel(x_ptr, out_ptr, n: tl.constexpr):
    offsets = tl.arange(0, n)
    tl.store(out_ptr + offsets, round_bf16_to_fp32(tl.load(x_ptr + offsets)))


def round_semantics() -> None:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(34351)
    size = 256
    x = torch.randn(size, device="cuda", dtype=torch.float32, generator=generator)
    x[0] = 1.0
    x[1] = -1.0
    x[2] = 1.0000001
    x[3] = -1.0000001
    x[4] = float("nan")
    x[5] = float("inf")
    out = torch.empty_like(x)
    round_bf16_kernel[(1,)](x, out, n=size)
    torch.cuda.synchronize()
    reference = x.to(torch.bfloat16).to(torch.float32)
    finite_reference = x[:4].to(torch.bfloat16).to(torch.float32)
    print(f"round_bf16_bitwise_equal_finite={torch.equal(out[:4], finite_reference)}")
    print(f"round_bf16_nan_mask_equal={torch.equal(torch.isnan(out), torch.isnan(reference))}")
    print(f"round_bf16_signed_zero_mask_equal={torch.equal(torch.signbit(out), torch.signbit(reference))}")
    print(f"round_bf16_inf_mask_equal={torch.equal(torch.isinf(out), torch.isinf(reference))}")
    print(f"round_bf16_input_head={x[:8].cpu().tolist()}")
    print(f"round_bf16_output_head={out[:8].cpu().tolist()}")
    print(f"round_bf16_reference_head={reference[:8].cpu().tolist()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["direct-ln", "direct-qk", "semantics", "round"])
    args = parser.parse_args()
    if args.mode == "direct-ln":
        direct_ln()
    elif args.mode == "direct-qk":
        direct_qk()
    elif args.mode == "semantics":
        semantics()
    else:
        round_semantics()


if __name__ == "__main__":
    main()
