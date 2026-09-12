# SPDX-License-Identifier: Apache-2.0
"""DeepSeek V3.2 (DSA) model-construction and pipeline helpers.

Keep the DSA-specific policy here while the concrete attention and model blocks
remain in ``deepseek_v2.py`` for import compatibility.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from sglang.srt.configs.model_config import (
    dsa_layer_skips_topk,
    get_dsa_index_head_dim,
    get_dsa_index_kpool,
    get_dsa_index_n_heads,
    get_dsa_index_topk,
    is_deepseek_dsa,
)
from sglang.srt.layers.attention.dsa.dsa_indexer import Indexer
from sglang.srt.layers.attention.dsa.dsa_indexer_kpool import IndexerKPool
from sglang.srt.utils.common import add_prefix


class DeepseekV32AttentionMixin:
    """Construct the optional DSA indexer used by DeepSeek V3.2 attention."""

    def init_dsa_indexer(
        self,
        *,
        config,
        hidden_size: int,
        qk_rope_head_dim: int,
        q_lora_rank: Optional[int],
        max_position_embeddings: int,
        rope_theta: float,
        rope_scaling: Optional[Dict[str, Any]],
        quant_config,
        layer_id: int,
        prefix: str,
        alt_stream: Optional[torch.cuda.Stream],
        skip_rope: bool,
        is_nextn: bool,
    ) -> None:
        self.use_dsa = is_deepseek_dsa(config)
        self.skip_topk = None
        self.next_skip_topk = None
        self.indexer = None
        if not self.use_dsa:
            return

        # NextN always owns an indexer. Normal skip-topk layers intentionally do
        # not: their checkpoint has no indexer weights and they reuse the
        # preceding layer's indices.
        if is_nextn:
            self.skip_topk = True
            self.next_skip_topk = True
        else:
            self.skip_topk = dsa_layer_skips_topk(config, layer_id)
            self.next_skip_topk = dsa_layer_skips_topk(config, layer_id + 1)

        if self.skip_topk and not is_nextn:
            return

        indexer_cls = IndexerKPool if get_dsa_index_kpool(config) > 1 else Indexer
        indexer_kwargs = dict(
            hidden_size=hidden_size,
            index_n_heads=get_dsa_index_n_heads(config),
            index_head_dim=get_dsa_index_head_dim(config),
            rope_head_dim=qk_rope_head_dim,
            index_topk=get_dsa_index_topk(config),
            q_lora_rank=q_lora_rank,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            scale_fmt="ue8m0",
            block_size=128,
            rope_scaling=rope_scaling,
            is_neox_style=not getattr(config, "indexer_rope_interleave", False),
            prefix=add_prefix("indexer", prefix),
            quant_config=quant_config,
            layer_id=layer_id,
            alt_stream=alt_stream,
            config=config,
        )
        if indexer_cls is IndexerKPool:
            indexer_kwargs["skip_rope"] = skip_rope
        self.indexer = indexer_cls(**indexer_kwargs)


class DeepseekV32ModelMixin:
    """DSA top-k routing policy shared by DeepSeek V3.2 model execution."""

    def init_dsa_model(self, config) -> None:
        self.use_dsa = is_deepseek_dsa(config)

    def _dsa_forward_uses_topk(self) -> bool:
        if not self.use_dsa:
            return False
        # Import lazily to keep this policy module independent from model-runner
        # initialization and easy to exercise with a mocked backend.
        from sglang.srt.model_executor.forward_context import get_attn_backend

        backend = get_attn_backend()
        backend = getattr(backend, "primary", backend)
        return not getattr(backend, "use_mha", False)

    def dsa_stage_requires_input_topk(self, dsa_forward_uses_topk: bool) -> bool:
        return (
            self.use_dsa
            and dsa_forward_uses_topk
            and dsa_layer_skips_topk(self.config, self.start_layer)
        )

    def dsa_stage_must_forward_topk(self, dsa_forward_uses_topk: bool) -> bool:
        return (
            self.use_dsa
            and dsa_forward_uses_topk
            and self.end_layer < self.config.num_hidden_layers
            and dsa_layer_skips_topk(self.config, self.end_layer)
        )

    def empty_dsa_topk(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states.new_empty(
            (0, get_dsa_index_topk(self.config)), dtype=torch.int32
        )
