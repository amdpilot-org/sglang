import argparse

import torch

from aiter.ops.triton.fp8_mqa_logits import fp8_mqa_logits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("size", type=int)
    args = parser.parse_args()

    size = args.size
    device = "cuda"
    dtype = torch.float8_e4m3fn
    q = torch.zeros((size, 32, 128), dtype=dtype, device=device)
    kv = torch.zeros((size, 128), dtype=dtype, device=device)
    scales = torch.ones(size, dtype=torch.float32, device=device)
    weights = torch.ones((size, 32), dtype=torch.float32, device=device)
    starts = torch.zeros(size, dtype=torch.int32, device=device)
    ends = torch.full((size,), size, dtype=torch.int32, device=device)

    logits = fp8_mqa_logits(
        q, kv, scales, weights, starts, ends, clean_logits=False
    )
    torch.cuda.synchronize()
    print(
        f"returned shape={tuple(logits.shape)} bytes={logits.numel() * logits.element_size()} "
        f"max_abs={logits.abs().max().item()}"
    )


if __name__ == "__main__":
    main()
