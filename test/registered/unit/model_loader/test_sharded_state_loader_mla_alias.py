import torch
from torch import nn

from sglang.srt.model_loader.loader import ShardedStateLoader


class _AttentionChild(nn.Module):
    def __init__(self):
        super().__init__()
        self.kv_b_proj = None


class _MLAAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.kv_b_proj = nn.Linear(2, 2, bias=False)
        self.attn_mha = _AttentionChild()

    def run_forward_setup(self):
        self.attn_mha.kv_b_proj = self.kv_b_proj


def test_post_forward_sharded_key_resolves_on_fresh_mla_model():
    saved_model = _MLAAttention()
    saved_model.run_forward_setup()
    saved_state = ShardedStateLoader._filter_subtensors(saved_model.state_dict())

    assert list(saved_state) == ["attn_mha.kv_b_proj.weight"]

    fresh_state = ShardedStateLoader._filter_subtensors(
        _MLAAttention().state_dict()
    )
    checkpoint_key = next(iter(saved_state))
    resolved_key = ShardedStateLoader._resolve_state_dict_key(
        f"model.layers.0.self_attn.{checkpoint_key}",
        {f"model.layers.0.self_attn.{key}": value for key, value in fresh_state.items()},
    )

    assert resolved_key == "model.layers.0.self_attn.kv_b_proj.weight"


def test_existing_checkpoint_key_is_not_remapped():
    alias_key = "model.layers.0.self_attn.attn_mha.kv_b_proj.weight"
    state_dict = {alias_key: torch.empty(1)}

    assert ShardedStateLoader._resolve_state_dict_key(alias_key, state_dict) == alias_key


def test_unrelated_attn_mha_key_is_not_remapped():
    checkpoint_key = "model.layers.0.self_attn.attn_mha.q_proj.weight"
    state_dict = {"model.layers.0.self_attn.q_proj.weight": torch.empty(1)}

    assert (
        ShardedStateLoader._resolve_state_dict_key(checkpoint_key, state_dict)
        == checkpoint_key
    )


def test_missing_canonical_mla_key_is_not_hidden():
    checkpoint_key = "model.layers.0.self_attn.attn_mha.kv_b_proj.weight"

    assert ShardedStateLoader._resolve_state_dict_key(checkpoint_key, {}) == checkpoint_key
