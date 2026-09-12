"""Exercise token-row preservation in a real ROCm graph and eager reference.

This is deliberately a backend-mechanics fixture, not a Qwen3.8 model reproduction.
"""

import json

import torch


def run_case(batch_size: int, draft_tokens: int) -> dict:
    rows = batch_size * draft_tokens
    static_input = torch.arange(rows * 7, device="cuda", dtype=torch.float32).reshape(
        rows, 7
    )
    eager = torch.softmax(static_input / 17.0, dim=-1)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = torch.softmax(static_input / 17.0, dim=-1)
    graph.replay()
    torch.cuda.synchronize()

    return {
        "batch_size": batch_size,
        "draft_tokens": draft_tokens,
        "expected_rows": rows,
        "captured_shape": list(captured.shape),
        "all_finite": bool(torch.isfinite(captured).all().item()),
        "nonnegative": bool((captured >= 0).all().item()),
        "row_sums_max_abs_error": float((captured.sum(-1) - 1).abs().max().item()),
        "eager_graph_max_abs_error": float((captured - eager).abs().max().item()),
    }


def main() -> None:
    torch.cuda.init()
    result = {
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "device": torch.cuda.get_device_name(0),
        "gfx_arch": torch.cuda.get_device_properties(0).gcnArchName,
        "cases": [run_case(bs, 4) for bs in (1, 2, 8)],
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
