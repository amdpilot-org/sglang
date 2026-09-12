# SPDX-License-Identifier: Apache-2.0
"""AMD attention dispatch for DeepSeek V2-family models."""

from sglang.srt.models.deepseek_common.attention_backend_handler import (
    AttnForwardMethod,
)
from sglang.srt.models.deepseek_common.attention_forward_methods import (
    DeepseekMHARocmForwardMixin,
    DeepseekMLAFusedRopeRocmForwardMixin,
    DeepseekMLARocmForwardMixin,
)


class DeepseekV2AMDAttentionMixin(
    DeepseekMHARocmForwardMixin,
    DeepseekMLARocmForwardMixin,
    DeepseekMLAFusedRopeRocmForwardMixin,
):
    """Own AMD-specific attention implementations and their dispatch."""

    _AMD_FORWARD_METHODS = frozenset(
        {
            AttnForwardMethod.MHA_ROCM,
            AttnForwardMethod.MHA_ONE_SHOT_ROCM,
            AttnForwardMethod.MHA_CHUNKED_KV_ROCM,
            AttnForwardMethod.MLA_ROCM,
            AttnForwardMethod.MLA_FUSED_ROPE_ROCM,
        }
    )

    def forward_amd_prepare(
        self,
        method,
        positions,
        hidden_states,
        forward_batch,
        zero_allocator,
        llama_4_scaling,
        prev_topk_indices,
    ):
        args = (positions, hidden_states, forward_batch, zero_allocator)
        if method == AttnForwardMethod.MHA_ROCM:
            return self.forward_normal_rocm_prepare(*args)
        if method == AttnForwardMethod.MHA_ONE_SHOT_ROCM:
            return self.forward_normal_one_shot_rocm_prepare(*args)
        if method == AttnForwardMethod.MHA_CHUNKED_KV_ROCM:
            return self.forward_normal_chunked_kv_rocm_prepare(*args)
        if method == AttnForwardMethod.MLA_ROCM:
            return self.forward_absorb_rocm_prepare(
                *args, llama_4_scaling, prev_topk_indices
            )
        if method == AttnForwardMethod.MLA_FUSED_ROPE_ROCM:
            return self.forward_absorb_fused_mla_rope_prepare(*args)
        raise NotImplementedError(method)

    def forward_amd_core(self, method, inner_state):
        if method in (
            AttnForwardMethod.MHA_ROCM,
            AttnForwardMethod.MHA_ONE_SHOT_ROCM,
            AttnForwardMethod.MHA_CHUNKED_KV_ROCM,
        ):
            return self.forward_normal_core(*inner_state)
        if method == AttnForwardMethod.MLA_ROCM:
            return self.forward_absorb_rocm_core(*inner_state)
        if method == AttnForwardMethod.MLA_FUSED_ROPE_ROCM:
            return self.forward_absorb_fused_mla_rope_core(*inner_state)
        raise NotImplementedError(method)
