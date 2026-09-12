from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend
from sglang.srt.runtime_context import get_platform
from sglang.srt.utils import is_hip

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner


class SageAttentionBackend(TorchNativeAttnBackend):
    """SageAttention over SGLang's gathered paged KV cache.

    SageAttention does not consume SGLang's paged cache directly.  The parent
    backend gathers each request into dense K/V tensors, then this class runs
    SageAttention's INT8 Q/K kernel.  CUDA graphs remain disabled for this
    correctness-first integration, as they are for ``torch_native``.
    """

    def __init__(self, model_runner: ModelRunner):
        if is_hip():
            raise RuntimeError(
                "The sage attention backend requires NVIDIA CUDA; "
                "the upstream SageAttention extension does not support ROCm."
            )
        if get_platform().is_sm100:
            raise RuntimeError(
                "The sage attention backend is not supported on SM100 by the "
                "pinned SageAttention revision: its public sageattn dispatcher "
                "raises 'Unsupported CUDA architecture: sm100'."
            )
        try:
            from sageattention import sageattn
        except ImportError as exc:
            raise ImportError(
                "The sage attention backend requires SageAttention. Install it "
                "from https://github.com/thu-ml/SageAttention before launching "
                "the server."
            ) from exc

        super().__init__(model_runner)
        self.sageattn = sageattn

    def _attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        attn_mask: Optional[torch.Tensor],
        enable_gqa: bool,
        scale: Optional[float],
        is_causal: bool,
    ) -> torch.Tensor:
        if attn_mask is not None:
            raise ValueError(
                "The sage attention backend does not support arbitrary or "
                "sliding-window attention masks."
            )
        if query.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(
                "The sage attention backend requires float16 or bfloat16 Q/K/V."
            )
        if query.shape[-1] not in (64, 128):
            raise ValueError(
                "The sage attention backend supports head dimensions 64 and 128."
            )

        # SageAttention's dense API expects equal Q and KV head counts. Preserve
        # GQA/MQA model semantics by materializing the same expansion performed
        # by PyTorch SDPA's enable_gqa path.
        if enable_gqa:
            if query.shape[1] % key.shape[1] != 0:
                raise ValueError("Query heads must be divisible by KV heads for GQA.")
            repeat = query.shape[1] // key.shape[1]
            key = key.repeat_interleave(repeat, dim=1)
            value = value.repeat_interleave(repeat, dim=1)

        return self.sageattn(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            tensor_layout="HND",
            is_causal=is_causal,
            sm_scale=scale,
        )
