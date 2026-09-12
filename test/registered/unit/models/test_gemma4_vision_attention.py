import torch
from torch import nn

from sglang.srt.models.gemma4_vision import Gemma4VisionAttention
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _IdentityQKV(nn.Module):
    def forward(self, hidden_states):
        return hidden_states, hidden_states, hidden_states


class _CapturingBackend:
    def __init__(self):
        self.calls = []

    def forward(self, q, k, v, **kwargs):
        self.calls.append(kwargs)
        return q


def _run_attention(batch_size: int, seq_len: int):
    attention = Gemma4VisionAttention.__new__(Gemma4VisionAttention)
    nn.Module.__init__(attention)
    attention.head_dim = 4
    attention.num_heads_per_partition = 1
    attention.num_kv_heads_per_partition = 1
    attention.qkv = _IdentityQKV()
    attention.q_norm = nn.Identity()
    attention.k_norm = nn.Identity()
    attention.v_norm = nn.Identity()
    attention.o_proj = nn.Identity()
    attention.qkv_backend = _CapturingBackend()

    hidden_states = torch.randn(batch_size, seq_len, 4)
    cos = torch.ones(batch_size, seq_len, 4)
    sin = torch.zeros_like(cos)
    output = attention(hidden_states, cos, sin)
    return output, attention.qkv_backend.calls


def test_gemma4_vision_passes_dense_max_seqlen_to_backend():
    output, calls = _run_attention(batch_size=1, seq_len=7)

    assert output.shape == (1, 7, 4)
    assert len(calls) == 1
    assert calls[0]["cu_seqlens"] is None
    assert calls[0]["max_seqlen"] == 7
    assert calls[0]["softmax_scale"] == 1.0


def test_gemma4_vision_batched_inputs_keep_per_item_max_seqlen():
    output, calls = _run_attention(batch_size=3, seq_len=5)

    assert output.shape == (3, 5, 4)
    assert calls[0]["bsz"] == 3
    assert calls[0]["max_seqlen"] == 5
    assert calls[0]["softmax_scale"] == 1.0
