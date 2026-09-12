# SPDX-License-Identifier: Apache-2.0
"""Hardware-specific DeepSeek model mixins."""

from .deepseek_v2_amd_mixin import DeepseekV2AMDAttentionMixin
from .deepseek_v2_cpu_mixin import DeepseekV2CPUAttentionMixin
from .deepseek_v2_npu_mixin import DeepseekV2NPUAttentionMixin

__all__ = [
    "DeepseekV2AMDAttentionMixin",
    "DeepseekV2CPUAttentionMixin",
    "DeepseekV2NPUAttentionMixin",
]
