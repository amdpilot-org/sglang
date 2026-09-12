# SPDX-License-Identifier: Apache-2.0
"""NPU attention dispatch for DeepSeek V2-family models.

The NPU implementation is imported lazily so importing ``deepseek_v2`` on a
non-NPU host does not require the NPU runtime.
"""

from sglang.srt.models.deepseek_common.attention_backend_handler import (
    AttnForwardMethod,
)


def _npu_attention_functions():
    from sglang.srt.hardware_backend.npu.modules.deepseek_v2_attention_mla_npu import (
        forward_dsa_core_npu,
        forward_dsa_prepare_npu,
        forward_mha_core_npu,
        forward_mha_prepare_npu,
        forward_mla_core_npu,
        forward_mla_prepare_npu,
    )

    return {
        AttnForwardMethod.MHA_NPU: (forward_mha_prepare_npu, forward_mha_core_npu),
        AttnForwardMethod.MLA_NPU: (forward_mla_prepare_npu, forward_mla_core_npu),
        AttnForwardMethod.DSA_NPU: (forward_dsa_prepare_npu, forward_dsa_core_npu),
    }


class DeepseekV2NPUAttentionMixin:
    """Route NPU-specific attention preparation and execution."""

    def forward_npu_prepare(
        self,
        attn_forward_method,
        positions,
        hidden_states,
        forward_batch,
        zero_allocator,
        layer_scatter_modes,
        prev_topk_indices,
    ):
        prepare, _ = _npu_attention_functions()[attn_forward_method]
        args = (
            self,
            positions,
            hidden_states,
            forward_batch,
            zero_allocator,
            layer_scatter_modes,
        )
        if attn_forward_method == AttnForwardMethod.DSA_NPU:
            args += (prev_topk_indices,)
        return prepare(*args)

    def forward_npu_core(self, attn_forward_method, inner_state):
        _, core = _npu_attention_functions()[attn_forward_method]
        return core(self, *inner_state)
