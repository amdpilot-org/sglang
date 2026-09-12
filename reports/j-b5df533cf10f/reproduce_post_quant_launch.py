import argparse

import torch

from sglang.kernels.ops.attention.dsv4.moe import (
    silu_and_mul_contig_post_quant,
    silu_and_mul_masked_post_quant,
)


def run_contig(tokens: int, hidden: int) -> None:
    x = torch.randn((tokens, hidden * 2), device="cuda", dtype=torch.bfloat16)
    out = torch.empty((tokens, hidden), device="cuda", dtype=torch.float8_e4m3fnuz)
    scale = torch.empty((tokens, hidden // 128), device="cuda", dtype=torch.float32)
    silu_and_mul_contig_post_quant(x, out, scale, 128)
    torch.cuda.synchronize()


def run_masked(tokens: int, hidden: int, topk: int) -> None:
    experts = 1
    x = torch.randn(
        (experts, tokens, hidden * 2), device="cuda", dtype=torch.bfloat16
    )
    out = torch.empty(
        (experts, tokens, hidden), device="cuda", dtype=torch.float8_e4m3fnuz
    )
    scale = torch.empty(
        (experts, tokens, hidden // 128), device="cuda", dtype=torch.float32
    )
    masked_m = torch.full((experts,), tokens, device="cuda", dtype=torch.int32)
    silu_and_mul_masked_post_quant(x, out, scale, 128, masked_m, topk=topk)
    torch.cuda.synchronize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("variant", choices=("contig", "masked"))
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--hidden", type=int, required=True)
    parser.add_argument("--topk", type=int, default=1)
    args = parser.parse_args()
    print(torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).gcnArchName)
    if args.variant == "contig":
        run_contig(args.tokens, args.hidden)
    else:
        run_masked(args.tokens, args.hidden, args.topk)
    print("completed")
