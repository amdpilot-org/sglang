from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.models.deepseek_common import v32_mixin
from sglang.srt.models.deepseek_v2 import (
    DeepseekV2AttentionMLA,
    DeepseekV2ForCausalLM,
    DeepseekV2Model,
)


class Attention(v32_mixin.DeepseekV32AttentionMixin):
    pass


class Model(v32_mixin.DeepseekV32ModelMixin):
    pass


assert issubclass(DeepseekV2AttentionMLA, v32_mixin.DeepseekV32AttentionMixin)
assert issubclass(DeepseekV2Model, v32_mixin.DeepseekV32ModelMixin)
assert DeepseekV2ForCausalLM.__module__ == "sglang.srt.models.deepseek_v2"

for skip, nextn, kpool in [(False, False, 1), (True, False, 1), (True, True, 1), (False, False, 2)]:
    obj = Attention()
    config = SimpleNamespace(indexer_rope_interleave=True)
    with (
        patch.object(v32_mixin, "is_deepseek_dsa", return_value=True),
        patch.object(v32_mixin, "dsa_layer_skips_topk", return_value=skip),
        patch.object(v32_mixin, "get_dsa_index_kpool", return_value=kpool),
        patch.object(v32_mixin, "get_dsa_index_n_heads", return_value=3),
        patch.object(v32_mixin, "get_dsa_index_head_dim", return_value=24),
        patch.object(v32_mixin, "get_dsa_index_topk", return_value=11),
        patch.object(v32_mixin, "Indexer") as plain,
        patch.object(v32_mixin, "IndexerKPool") as pooled,
    ):
        obj.init_dsa_indexer(config=config, hidden_size=64, qk_rope_head_dim=8,
            q_lora_rank=16, max_position_embeddings=1024, rope_theta=10000,
            rope_scaling=None, quant_config=None, layer_id=2, prefix="x",
            alt_stream=None, skip_rope=True, is_nextn=nextn)
    expected_construct = (not skip) or nextn
    assert plain.call_count + pooled.call_count == int(expected_construct)
    if expected_construct:
        call = pooled if kpool > 1 else plain
        assert call.call_args.kwargs["index_topk"] == 11
        assert call.call_args.kwargs["is_neox_style"] is False
        assert ("skip_rope" in call.call_args.kwargs) == (kpool > 1)

model = Model()
model.use_dsa = True
model.start_layer = 2
model.end_layer = 5
model.config = SimpleNamespace(num_hidden_layers=6)
for active, start_skip, end_skip in [(False, True, True), (True, False, False), (True, True, False), (True, False, True)]:
    with patch.object(v32_mixin, "dsa_layer_skips_topk", side_effect=lambda _, layer: start_skip if layer == 2 else end_skip):
        assert model.dsa_stage_requires_input_topk(active) == (active and start_skip)
        assert model.dsa_stage_must_forward_topk(active) == (active and end_skip)

device = "cuda" if torch.cuda.is_available() else "cpu"
x = torch.randn(3, 7, device=device)
with patch.object(v32_mixin, "get_dsa_index_topk", return_value=13):
    empty = model.empty_dsa_topk(x)
assert empty.shape == (0, 13) and empty.dtype == torch.int32 and empty.device == x.device
print({"status": "ok", "device": str(x.device), "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None})
