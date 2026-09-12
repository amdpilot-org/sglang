"""Shared bf16-input/fp32-output GEMM dispatcher."""

import functools
import importlib.util
from typing import Optional

import torch

from sglang.srt.environ import envs

_linear_bf16_fp32_algo = envs.SGLANG_OPT_BF16_FP32_GEMM_ALGO.get()
_HPC_GEMM_WEIGHT_CACHE_ATTR = "_sglang_bf16xfp32_weight_cache"
_HPC_GEMM_WEIGHT_SCALE = 1.0 / 256.0
_hpc_gemm_enabled = False


@functools.cache
def _hpc_gemm_bf16xfp32_available() -> bool:
    """HPC-Ops (https://github.com/Tencent/hpc-ops) ships sm90a kernels."""
    if importlib.util.find_spec("hpc") is None or not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major == 9


def _can_use_hpc_gemm_bf16xfp32(
    x: torch.Tensor, y: torch.Tensor, *, min_m: int = 8
) -> bool:
    return (
        x.dim() == 2
        and y.dim() == 2
        and x.shape[1] == y.shape[1]
        and x.shape[0] >= min_m
        and x.is_cuda
        and y.is_cuda
        and x.dtype == torch.bfloat16
        and y.dtype == torch.float32
        and x.is_contiguous()
        and y.is_contiguous()
        and y.shape[0] % 64 == 0
        and _hpc_gemm_bf16xfp32_available()
    )


def _get_bf16xfp32_weight_split(
    y: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return and cache the HPC-Ops high/low bf16 weight decomposition."""
    import hpc

    if not hpc_bf16xfp32_gemm_enabled():
        raise RuntimeError(
            "Call mark_hpc_bf16xfp32_gemm_enabled() at model init before "
            "routing GEMMs to the HPC-Ops bf16xfp32 kernel."
        )
    cache_key = (
        y.data_ptr(),
        tuple(y.shape),
        tuple(y.stride()),
        y.device.index,
        y.dtype,
    )
    cache = getattr(y, _HPC_GEMM_WEIGHT_CACHE_ATTR, None)
    if cache is not None and cache[0] == cache_key:
        return cache[1], cache[2], cache[3]
    with torch.no_grad():
        w_high = y.to(torch.bfloat16)
        w_low = ((y - w_high.float()) / _HPC_GEMM_WEIGHT_SCALE).to(torch.bfloat16)
    split_flag = hpc.get_gemm_bf16xfp32_workspace(y.shape[0])
    setattr(y, _HPC_GEMM_WEIGHT_CACHE_ATTR, (cache_key, w_high, w_low, split_flag))
    return w_high, w_low, split_flag


def mark_hpc_bf16xfp32_gemm_enabled() -> None:
    """Enable startup-safe caching of HPC-Ops split weights when available."""
    global _hpc_gemm_enabled
    if _hpc_gemm_bf16xfp32_available():
        _hpc_gemm_enabled = True


def hpc_bf16xfp32_gemm_enabled() -> bool:
    if _hpc_gemm_enabled:
        return True
    return _linear_bf16_fp32_algo == "hpc" and _hpc_gemm_bf16xfp32_available()


def _linear_bf16_fp32_cublas(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if x.is_cuda and x.dtype == torch.bfloat16 and y.dtype == torch.bfloat16:
        return torch.mm(x, y.t(), out_dtype=torch.float32)
    return torch.mm(x.float(), y.float().t())


def _linear_bf16_fp32_hpc(
    x: torch.Tensor, y: torch.Tensor, *, min_m: int = 8
) -> Optional[torch.Tensor]:
    if not _can_use_hpc_gemm_bf16xfp32(x, y, min_m=min_m):
        return None
    import hpc

    w_high, w_low, split_flag = _get_bf16xfp32_weight_split(y)
    return hpc.gemm_bf16xfp32(
        x,
        w_high,
        w_low,
        _HPC_GEMM_WEIGHT_SCALE,
        use_fp32_output=True,
        use_splitk=True,
        split_flag=split_flag,
    )


def linear_bf16_fp32(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    hpc_kernel_min_m: Optional[int] = None,
) -> torch.Tensor:
    if hpc_kernel_min_m is not None:
        output = _linear_bf16_fp32_hpc(x, y, min_m=hpc_kernel_min_m)
        return output if output is not None else _linear_bf16_fp32_cublas(x, y)
    if _linear_bf16_fp32_algo == "hpc":
        output = _linear_bf16_fp32_hpc(x, y)
        return output if output is not None else _linear_bf16_fp32_cublas(x, y)
    if _linear_bf16_fp32_algo == "deep_gemm" and y.dtype == torch.bfloat16:
        from sglang.srt.layers import deep_gemm_wrapper

        output = torch.empty(x.size(0), y.size(0), dtype=torch.float32, device=x.device)
        deep_gemm_wrapper.gemm_nt_bf16bf16f32(x, y, output)
        return output
    return _linear_bf16_fp32_cublas(x, y)


__all__ = [
    "hpc_bf16xfp32_gemm_enabled",
    "linear_bf16_fp32",
    "mark_hpc_bf16xfp32_gemm_enabled",
]
