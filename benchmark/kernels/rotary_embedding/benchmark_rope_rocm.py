import argparse
import json
from pathlib import Path

import torch

from sglang.srt.layers.rotary_embedding import RotaryEmbedding
from sglang.srt.utils import is_hip


POSITIONS = (0, 1, 7, 63, 255, 1023, 4095, 8191, 16383)
HEAD_DIMS = (
    (64, 32),
    (64, 64),
    (128, 64),
    (128, 128),
    (256, 128),
    (256, 256),
    (512, 256),
    (512, 512),
)
DTYPES = (torch.float16, torch.bfloat16)


def bounded_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if not 1 <= parsed <= 10000:
        raise argparse.ArgumentTypeError("must be between 1 and 10000")
    return parsed


def measure_latency(
    rope: RotaryEmbedding,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    warmup_iterations: int,
    timed_iterations: int,
) -> float:
    for _ in range(warmup_iterations):
        rope.forward_cuda(positions, query, key)
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(timed_iterations):
        rope.forward_cuda(positions, query, key)
    end_event.record()
    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event) * 1000.0 / timed_iterations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=bounded_integer, default=10)
    parser.add_argument("--iterations", type=bounded_integer, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not (is_hip() and torch.cuda.is_available()):
        raise RuntimeError("Requires an AMD HIP GPU and the native rotary kernel")

    torch.manual_seed(35003)
    device = torch.device("cuda")
    positions = torch.tensor(POSITIONS, dtype=torch.int64, device=device)
    results = []
    for dtype in DTYPES:
        for head_size, rotary_dim in HEAD_DIMS:
            for is_neox in (True, False):
                rope = RotaryEmbedding(
                    head_size=head_size,
                    rotary_dim=rotary_dim,
                    max_position_embeddings=max(POSITIONS) + 1,
                    base=10000,
                    is_neox_style=is_neox,
                    dtype=dtype,
                ).to(device)
                query = torch.randn(
                    positions.numel(), 2 * head_size, dtype=dtype, device=device
                )
                key = torch.randn(
                    positions.numel(), head_size, dtype=dtype, device=device
                )
                latency_us = measure_latency(
                    rope,
                    positions,
                    query,
                    key,
                    args.warmup,
                    args.iterations,
                )
                results.append(
                    {
                        "dtype": str(dtype).removeprefix("torch."),
                        "head_size": head_size,
                        "rotary_dim": rotary_dim,
                        "rotary_fraction": rotary_dim / head_size,
                        "is_neox": is_neox,
                        "positions": list(POSITIONS),
                        "warm_latency_us": latency_us,
                        "cache_dtype": str(rope.cos_sin_cache.dtype).removeprefix(
                            "torch."
                        ),
                    }
                )

    report = {
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "hip_version": torch.version.hip,
        "warmup_iterations": args.warmup,
        "timed_iterations": args.iterations,
        "timing_method": "CUDA events around the requested number of forward_cuda calls",
        "cases": results,
    }
    rendered = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
