"""Minimal reproduction of the MLA sharded-state key transition."""

import torch
from torch import nn

from sglang.srt.model_loader.loader import ShardedStateLoader


class AttentionChild(nn.Module):
    def __init__(self):
        super().__init__()
        self.kv_b_proj = None


class MLAAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.kv_b_proj = nn.Linear(2, 2, bias=False)
        self.attn_mha = AttentionChild()


class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = MLAAttention()


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = Layer()


fresh = Model()
print("fresh filtered keys:", list(ShardedStateLoader._filter_subtensors(fresh.state_dict())))

warmed = Model()
warmed.layer.self_attn.attn_mha.kv_b_proj = warmed.layer.self_attn.kv_b_proj
saved = ShardedStateLoader._filter_subtensors(warmed.state_dict())
print("post-forward filtered keys:", list(saved))

checkpoint_key = next(iter(saved))
try:
    fresh.state_dict()[checkpoint_key]
except KeyError as exc:
    print("fresh direct lookup:", type(exc).__name__, repr(exc.args[0]))

resolved = ShardedStateLoader._resolve_state_dict_key(checkpoint_key, fresh.state_dict())
print("resolved key:", resolved)
assert resolved == "layer.self_attn.kv_b_proj.weight"
