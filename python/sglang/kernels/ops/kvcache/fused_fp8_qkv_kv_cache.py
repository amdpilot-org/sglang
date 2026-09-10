from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)
from sglang.kernels.ops.quantization.fp8_kernel import is_fp8_fnuz

if TYPE_CHECKING:
    from tvm_ffi.module import Module


FP8_DTYPE = torch.float8_e4m3fnuz if is_fp8_fnuz() else torch.float8_e4m3fn
_IS_HIP = torch.version.hip is not None

@cache_once
def _jit_fused_fp8_qkv_kv_cache_module(dtype: torch.dtype, use_pdl: bool) -> Module:
    args = make_cpp_args(dtype, use_pdl)
    return load_jit(
        "fused_fp8_qkv_kv_cache",
        *args,
        cuda_files=["attention/fused_fp8_qkv_kv_cache.cuh"],
        cuda_wrappers=[("fused_fp8_qkv_kv_cache", f"FusedFp8QkvKvCache<{args}>::run")],
    )


def _scale_to_f32(scale: Optional[torch.Tensor], device: torch.device) -> torch.Tensor:
    if scale is None:
        return torch.ones(1, dtype=torch.float32, device=device)
    return scale.to(torch.float32).reshape(1)


def fused_fp8_qkv_kv_cache(
    q: torch.Tensor | None,
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_loc: torch.Tensor,
    k_scale: Optional[torch.Tensor] = None,
    v_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor | None:
    """Fused FP8 quant of K/V (+ optional Q) + paged KV-cache write."""
    if k.dtype not in (torch.bfloat16, torch.float16):
        raise RuntimeError(f"Unsupported dtype {k.dtype}. Supported: bfloat16, float16")

    num_tokens = k.shape[0]
    k2 = k.reshape(num_tokens, -1)
    v2 = v.reshape(num_tokens, -1)
    kv_dim = k2.shape[1]

    k_cache2 = k_cache.view(-1, kv_dim)
    v_cache2 = v_cache.view(-1, kv_dim)
    if _IS_HIP:
        k_cache2 = k_cache2.view(torch.uint8)
        v_cache2 = v_cache2.view(torch.uint8)

    ks = _scale_to_f32(k_scale, k.device)
    vs = _scale_to_f32(v_scale, k.device)

    q2 = None
    q_out = None
    if q is not None:
        q2 = q.reshape(num_tokens, -1)
        q_out = torch.empty(q2.shape, dtype=FP8_DTYPE, device=q.device)

    module = _jit_fused_fp8_qkv_kv_cache_module(k.dtype, is_arch_support_pdl())
    q_native = q_out.view(torch.uint8) if q_out is not None and _IS_HIP else q_out
    module.fused_fp8_qkv_kv_cache(
        q2,
        k2,
        v2,
        q_native,
        k_cache2,
        v_cache2,
        cache_loc,
        ks,
        vs,
    )
    return q_out
