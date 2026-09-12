import argparse

import torch
import torch.nn.functional as F

from sglang.srt.layers.attention.vision import VisionTritonAttention
from sglang.srt.runtime_context import get_parallel


def run(batch: int, seq_len: int, pass_scale_to_forward: bool) -> None:
    heads, dim = 2, 32
    torch.manual_seed(batch * 100 + seq_len)
    q = torch.randn(batch * seq_len, heads, dim, device="cuda", dtype=torch.float16)
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

    forward_kwargs = {"softmax_scale": 1.0} if pass_scale_to_forward else {}
    got = backend.forward(
        q,
        k,
        v,
        cu_seqlens=None,
        bsz=batch,
        seq_len=seq_len,
        max_seqlen=seq_len,
        **forward_kwargs,
    )
    q4 = q.reshape(batch, seq_len, heads, dim).permute(0, 2, 1, 3)
    k4 = k.reshape(batch, seq_len, heads, dim).permute(0, 2, 1, 3)
    v4 = v.reshape(batch, seq_len, heads, dim).permute(0, 2, 1, 3)
    ref = F.scaled_dot_product_attention(q4, k4, v4, scale=1.0)
    ref = ref.permute(0, 2, 1, 3).reshape_as(got)
    print(f"shape=({batch},{seq_len}) pass_scale_to_forward={pass_scale_to_forward}")
    print(f"max_abs_error={(got.float() - ref.float()).abs().max().item():.8f}")
    print(f"device={torch.cuda.get_device_name(0)}")
    print(f"arch={torch.cuda.get_device_properties(0).gcnArchName}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--pass-scale-to-forward", action="store_true")
    args = parser.parse_args()
    run(args.batch, args.seq_len, args.pass_scale_to_forward)
