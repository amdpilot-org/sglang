import sys
import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.layers.attention.vision import VisionTritonAttention
from sglang.srt.models.gemma4_vision import Gemma4VisionAttention
from sglang.srt.runtime_context import get_parallel


class IdentityQKV(nn.Module):
    def forward(self, x):
        return x, x, x


class Capture:
    def __init__(self):
        self.kwargs = None

    def forward(self, q, k, v, **kwargs):
        self.kwargs = kwargs
        return q


def forwarding_case(batch, seq, mask):
    attn = Gemma4VisionAttention.__new__(Gemma4VisionAttention)
    nn.Module.__init__(attn)
    attn.head_dim = 4
    attn.num_heads_per_partition = 1
    attn.num_kv_heads_per_partition = 1
    attn.qkv = IdentityQKV()
    attn.q_norm = nn.Identity()
    attn.k_norm = nn.Identity()
    attn.v_norm = nn.Identity()
    attn.o_proj = nn.Identity()
    attn.qkv_backend = Capture()
    x = torch.randn(batch, seq, 4)
    cos = torch.ones_like(x)
    sin = torch.zeros_like(x)
    out = attn(x, cos, sin, attention_mask=mask)
    assert out.shape == x.shape
    assert attn.qkv_backend.kwargs["max_seqlen"] == seq
    print("forward", batch, seq, "masked", mask is not None, "PASS")


def gpu_case(batch, seq, heads=2, dim=32):
    torch.manual_seed(batch * 100 + seq)
    q = torch.randn(batch * seq, heads, dim, device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    with get_parallel().override(attn_tp_size=1):
        backend = VisionTritonAttention(
            head_dim=dim,
            num_heads=heads,
            num_kv_heads=heads,
            dropout=0.0,
            flatten_batch=True,
            softmax_in_single_precision=False,
            softmax_scale=1.0,
        )
    got = backend.forward(q, k, v, cu_seqlens=None, bsz=batch, seq_len=seq, max_seqlen=seq)
    q4 = q.reshape(batch, seq, heads, dim).permute(0, 2, 1, 3)
    k4 = k.reshape(batch, seq, heads, dim).permute(0, 2, 1, 3)
    v4 = v.reshape(batch, seq, heads, dim).permute(0, 2, 1, 3)
    ref = F.scaled_dot_product_attention(q4, k4, v4, scale=1.0)
    ref = ref.permute(0, 2, 1, 3).reshape_as(got)
    err = (got.float() - ref.float()).abs().max().item()
    assert err < 0.02, err
    print("gpu", batch, seq, "max_abs", err, "PASS")


mode = sys.argv[1]
if mode == "forward":
    forwarding_case(2, 1, torch.tensor([[1], [0]], dtype=torch.float32))
    forwarding_case(4, 9, torch.ones(4, 9))
elif mode == "gpu_seq1":
    gpu_case(2, 1)
elif mode == "gpu_batch4":
    gpu_case(4, 9)
elif mode == "gpu_reported_shape":
    gpu_case(3, 5)
else:
    raise ValueError(mode)
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).gcnArchName)
