"""Shared precision and dispatch policy for MoE router projections."""

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.environ import envs
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.utils import is_cuda, is_npu


def tiny_router_gemm_max_tokens(
    *, num_experts: int, hidden_size: int, weight_dtype: torch.dtype
) -> int:
    """Return the shared tiny-router cutoff, or -1 when unsupported."""
    if not is_cuda() or weight_dtype != torch.bfloat16:
        return -1
    device = torch.cuda.current_device()
    major, _ = torch.cuda.get_device_capability(device)
    if major < 9:
        return -1
    from sglang.kernels.ops.gemm.tiny_gemm import can_use_tiny_gemm

    return 16 if can_use_tiny_gemm(num_experts, hidden_size, max_m=16) else -1


def router_linear_bf16_fp32(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    *,
    tiny_max_tokens: int = -1,
) -> torch.Tensor:
    """bf16 router GEMM with an fp32 output and batch-invariant policy."""
    deterministic = envs.SGLANG_ENABLE_DETERMINISTIC_INFERENCE.get()
    if is_npu():
        # NPU does not implement aten::mm.dtype. Keep its established bf16
        # kernel while normalizing the logits to the shared fp32 API.
        return F.linear(hidden_states, weight).float()
    if not deterministic and hidden_states.is_cuda:
        try:
            from sglang.kernels.ops.gemm.router_gemv import (
                router_gemv,
                router_gemv_supported,
            )

            if router_gemv_supported(hidden_states, weight):
                output = router_gemv(hidden_states, weight)
                if output.dtype == torch.float32:
                    return output
        except ImportError:
            pass
    if (
        not deterministic
        and 0 < hidden_states.shape[0] <= tiny_max_tokens
        and hidden_states.is_cuda
    ):
        from sglang.kernels.ops.gemm.tiny_gemm import tiny_gemm_bf16

        return tiny_gemm_bf16(
            hidden_states, weight, out_dtype=torch.float32, max_m=tiny_max_tokens
        )

    if hidden_states.is_cuda:
        from sglang.kernels.ops.gemm.bf16_fp32 import linear_bf16_fp32

        return linear_bf16_fp32(hidden_states, weight)
    return F.linear(hidden_states.float(), weight.float())


class RouterGate(nn.Module):
    """Checkpoint-compatible MoE gate which always returns fp32 logits.

    ``fp32_compute`` preserves models whose checkpoints require fp32 inputs and
    weights. Validated bf16 routers use the cheaper bf16 GEMM with fp32 output.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        *,
        fp32_compute: bool = True,
        params_dtype: Optional[torch.dtype] = None,
        has_correction_bias: bool = False,
        correction_bias_shape: Optional[tuple[int, ...]] = None,
    ):
        super().__init__()
        weight_dtype = (
            torch.float32
            if fp32_compute
            else (params_dtype or torch.get_default_dtype())
        )
        self.fp32_compute = fp32_compute
        self.weight = nn.Parameter(
            torch.empty((num_experts, hidden_size), dtype=weight_dtype)
        )
        self.weight.weight_loader = default_weight_loader
        if has_correction_bias:
            shape = correction_bias_shape or (num_experts,)
            self.e_score_correction_bias = nn.Parameter(
                torch.empty(shape, dtype=torch.float32)
            )
        else:
            self.e_score_correction_bias = None
        self.tiny_router_gemm_max_tokens = tiny_router_gemm_max_tokens(
            num_experts=num_experts,
            hidden_size=hidden_size,
            weight_dtype=self.weight.dtype,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.fp32_compute or self.weight.dtype == torch.float32:
            return F.linear(hidden_states.float(), self.weight.float())
        return router_linear_bf16_fp32(
            hidden_states.to(torch.bfloat16),
            self.weight,
            tiny_max_tokens=self.tiny_router_gemm_max_tokens,
        )
