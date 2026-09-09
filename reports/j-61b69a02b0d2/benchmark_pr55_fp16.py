import json
import os
import sys
from types import SimpleNamespace

import torch


CHECKOUT = os.environ.get("SGLANG_BENCHMARK_CHECKOUT")
if not CHECKOUT:
    raise SystemExit("SGLANG_BENCHMARK_CHECKOUT is required")
sys.path.insert(0, os.path.join(CHECKOUT, "python"))

from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool  # noqa: E402


def build_pool(head_dim, v_head_dim):
    return MHATokenToKVPool(
        size=1023,
        page_size=1,
        dtype=torch.float8_e4m3fnuz,
        head_num=8,
        head_dim=head_dim,
        v_head_dim=v_head_dim,
        layer_num=1,
        device="cuda",
        enable_memory_saver=False,
        enable_alt_stream=False,
    )


def measure(pool, k, v, loc, k_scale, v_scale, warmup=30, iterations=200):
    layer = SimpleNamespace(layer_id=0)
    for _ in range(warmup):
        pool.set_kv_buffer(
            layer, loc, k, v, k_scale=k_scale, v_scale=v_scale
        )
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        pool.set_kv_buffer(
            layer, loc, k, v, k_scale=k_scale, v_scale=v_scale
        )
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / iterations


def main():
    torch.manual_seed(55)
    configs = [
        ("128x8x128_scalar", 128, 128, 128, 1.25, 1.75),
        ("128x8x128_device", 128, 128, 128, None, None),
        ("64x8x192x128_scalar", 192, 128, 64, 1.5, 2.5),
    ]
    results = []
    for name, head_dim, v_head_dim, tokens, scalar_k, scalar_v in configs:
        pool = build_pool(head_dim, v_head_dim)
        k = torch.randn(
            tokens, 8, head_dim, device="cuda", dtype=torch.float16
        )
        v = torch.randn(
            tokens, 8, v_head_dim, device="cuda", dtype=torch.float16
        )
        loc = torch.randint(
            1, pool.size + 1, (tokens,), device="cuda", dtype=torch.int64
        )
        if name.endswith("device"):
            k_scale = torch.tensor(1.25, device="cuda", dtype=torch.float32)
            v_scale = torch.tensor(1.75, device="cuda", dtype=torch.float32)
        else:
            k_scale = scalar_k
            v_scale = scalar_v
        latency_us = measure(pool, k, v, loc, k_scale, v_scale)
        result = {
            "config": name,
            "tokens": tokens,
            "head_dim": head_dim,
            "v_head_dim": v_head_dim,
            "scale": "device" if name.endswith("device") else "scalar",
            "latency_us": latency_us,
        }
        results.append(result)
        print(json.dumps(result, sort_keys=True))
    output = os.environ.get("SGLANG_BENCHMARK_OUTPUT")
    if output:
        with open(output, "w") as handle:
            json.dump(
                {
                    "checkout": CHECKOUT,
                    "torch_version": torch.__version__,
                    "hip_version": torch.version.hip,
                    "gpu": torch.cuda.get_device_name(0),
                    "visible_gpus": torch.cuda.device_count(),
                    "results": results,
                },
                handle,
                indent=2,
                sort_keys=True,
            )


if __name__ == "__main__":
    main()
