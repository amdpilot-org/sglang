# SPDX-License-Identifier: Apache-2.0
"""CPU attention dispatch for DeepSeek V2-family models."""

from sglang.srt.models.deepseek_common.attention_backend_handler import (
    AttnForwardMethod,
)
from sglang.srt.models.deepseek_common.attention_forward_methods import (
    DeepseekMLACpuForwardMixin,
)


class DeepseekV2CPUAttentionMixin(DeepseekMLACpuForwardMixin):
    """Own CPU-specific attention implementations and their dispatch."""

    _CPU_FORWARD_METHODS = frozenset({AttnForwardMethod.MLA_FUSED_ROPE_CPU})

    def forward_cpu_prepare(
        self, method, positions, hidden_states, forward_batch, zero_allocator
    ):
        if method == AttnForwardMethod.MLA_FUSED_ROPE_CPU:
            return self.forward_absorb_fused_mla_rope_cpu_prepare(
                positions, hidden_states, forward_batch, zero_allocator
            )
        raise NotImplementedError(method)

    def forward_cpu_core(self, method, inner_state):
        if method == AttnForwardMethod.MLA_FUSED_ROPE_CPU:
            return self.forward_absorb_fused_mla_rope_cpu_core(*inner_state)
        raise NotImplementedError(method)
