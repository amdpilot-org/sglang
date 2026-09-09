import argparse

import torch
import triton
import triton.language as tl

try:
    from sglang.kernels.ops.diffusion.common.numerics import (
        cuda_rsqrtf,
        div_rn_f32,
        mul_rn_f32,
        rsqrt_approx_f32,
    )
except ImportError:
    from sglang.kernels.ops.diffusion.triton.numerics import (
        cuda_rsqrtf,
        div_rn_f32,
        mul_rn_f32,
        rsqrt_approx_f32,
    )

try:
    from sglang.kernels.kda_kernels.layernorm_modulate_triton import _rcp4
except ImportError:
    try:
        from sglang.kernels.ops.diffusion.norm.layernorm_modulate_triton import _rcp4
    except ImportError:
        from sglang.kernels.ops.diffusion.triton.layernorm_modulate import _rcp4


HELPERS = {
    "mul_rn_f32": mul_rn_f32,
    "div_rn_f32": div_rn_f32,
    "rsqrt_approx_f32": rsqrt_approx_f32,
    "cuda_rsqrtf": cuda_rsqrtf,
    "_rcp4": _rcp4,
}


@triton.jit
def one_input_kernel(x_ptr, y_ptr, out_ptr, n: tl.constexpr, helper: tl.constexpr):
    offsets = tl.arange(0, n)
    x = tl.load(x_ptr + offsets)
    y = tl.load(y_ptr + offsets)
    if helper == 0:
        result = mul_rn_f32(x, y)
    elif helper == 1:
        result = div_rn_f32(x, y)
    elif helper == 2:
        result = rsqrt_approx_f32(x)
    elif helper == 3:
        result = cuda_rsqrtf(x)
    else:
        result = _rcp4(x)
    tl.store(out_ptr + offsets, result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("helper", choices=sorted(HELPERS))
    args = parser.parse_args()

    generator = torch.Generator(device="cuda")
    generator.manual_seed(34351)
    size = 256
    x = torch.rand(size, device="cuda", generator=generator) + 0.5
    y = torch.rand(size, device="cuda", generator=generator) + 0.5
    if args.helper == "_rcp4":
        x = torch.arange(1, size + 1, device="cuda", dtype=torch.float32)
    out = torch.empty(size, device="cuda", dtype=torch.float32)
    helper_index = sorted(HELPERS).index(args.helper)
    one_input_kernel[(1,)](x, y, out, n=size, helper=helper_index)
    torch.cuda.synchronize()
    print(f"helper={args.helper}")
    print(f"input_head={x[:8].cpu().tolist()}")
    print(f"output_head={out[:8].cpu().tolist()}")
    print(f"output_finite={bool(torch.isfinite(out).all().item())}")


if __name__ == "__main__":
    main()
