import torch
from aiter.ops.triton.fused_kv_cache import fused_qk_rope_cat_and_cache_mla
from aiter.ops.triton.fused_qk_concat import fused_qk_rope_cat

__all__ = ["fused_qk_rope_cat", "fused_qk_rope_cat_and_cache_mla"]


def aiter_dsv3_router_gemm(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
):
    """Compute router logits with fp32 accumulation and output on ROCm.

    Aiter's untuned torch fallback computes ``F.linear`` in bf16 and only casts
    its result to ``otype`` afterward. ROCm's ``torch.mm`` ``out_dtype`` path
    retains fp32 logit precision for bf16 inputs, matching the other
    ``MoEGate`` GEMM branches.
    """
    return torch.mm(
        hidden_states, weight.detach().transpose(0, 1), out_dtype=torch.float32
    )


def get_dsv3_gemm_output_zero_allocator_size(
    n_routed_experts: int, num_moe_layers: int, allocate_size: int, embedding_dim: int
):
    if embedding_dim != 7168 or n_routed_experts != 256:
        return 0

    per_layer_size = 256 * (allocate_size + n_routed_experts)

    return num_moe_layers * per_layer_size
