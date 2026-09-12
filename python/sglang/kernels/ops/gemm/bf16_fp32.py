"""Neutral home for the shared bf16-input/fp32-output GEMM dispatcher."""

# Compatibility implementation remains importable from its former DSv4 path.
# Keeping one function object also preserves its process-wide HPC weight cache.
from sglang.kernels.ops.attention.dsv4.gemm import (  # noqa: F401
    hpc_bf16xfp32_gemm_enabled,
    linear_bf16_fp32,
    mark_hpc_bf16xfp32_gemm_enabled,
)

__all__ = [
    "hpc_bf16xfp32_gemm_enabled",
    "linear_bf16_fp32",
    "mark_hpc_bf16xfp32_gemm_enabled",
]
