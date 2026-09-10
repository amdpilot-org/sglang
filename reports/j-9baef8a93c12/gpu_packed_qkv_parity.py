import argparse
import json
import os
import time

import torch


CASES = (
    {"name": "mha-small", "tokens": 6, "hidden": 32, "heads": 4, "kv_heads": 4, "head_dim": 8},
    {"name": "mha-64", "tokens": 1023, "hidden": 512, "heads": 16, "kv_heads": 16, "head_dim": 64},
    {"name": "gqa-64", "tokens": 1023, "hidden": 512, "heads": 16, "kv_heads": 4, "head_dim": 64},
    {"name": "gqa-72", "tokens": 1023, "hidden": 576, "heads": 16, "kv_heads": 4, "head_dim": 72},
    {"name": "gqa-80", "tokens": 1023, "hidden": 640, "heads": 16, "kv_heads": 4, "head_dim": 80},
    {"name": "gqa-128", "tokens": 1023, "hidden": 1024, "heads": 16, "kv_heads": 4, "head_dim": 128},
)


def packed_projection(x, weight, heads, kv_heads, head_dim):
    packed = torch.nn.functional.linear(x, weight)
    q_size = heads * head_dim
    kv_size = kv_heads * head_dim
    q, k, v = packed.split([q_size, kv_size, kv_size], dim=-1)
    views = (
        q.reshape(x.shape[0], heads, head_dim),
        k.reshape(x.shape[0], kv_heads, head_dim),
        v.reshape(x.shape[0], kv_heads, head_dim),
    )
    return packed, views


def independent_projection(x, weight, heads, kv_heads, head_dim):
    q_size = heads * head_dim
    kv_size = kv_heads * head_dim
    return (
        torch.nn.functional.linear(x, weight[:q_size]),
        torch.nn.functional.linear(x, weight[q_size : q_size + kv_size]),
        torch.nn.functional.linear(x, weight[q_size + kv_size :]),
    )


def check_case(case, dtype):
    torch.manual_seed(0)
    device = torch.device("cuda")
    x = torch.randn(case["tokens"], case["hidden"], device=device, dtype=dtype)
    output_rows = (case["heads"] + 2 * case["kv_heads"]) * case["head_dim"]
    weight = torch.randn(output_rows, case["hidden"], device=device, dtype=dtype) * 0.05
    packed, views = packed_projection(
        x, weight, case["heads"], case["kv_heads"], case["head_dim"]
    )
    references = independent_projection(
        x, weight, case["heads"], case["kv_heads"], case["head_dim"]
    )
    expected_views = tuple(
        value.reshape(x.shape[0], heads, case["head_dim"])
        for value, heads in zip(references, (case["heads"], case["kv_heads"], case["kv_heads"]))
    )
    differences = [
        (actual.float() - reference.float()).abs().max().item()
        for actual, reference in zip(views, expected_views)
    ]
    packed_pointer = packed.untyped_storage().data_ptr()
    q_size = case["heads"] * case["head_dim"]
    kv_size = case["kv_heads"] * case["head_dim"]
    result = {
        **case,
        "dtype": str(dtype).removeprefix("torch."),
        "max_abs_diff": max(differences),
        "bit_identical": all(
            torch.equal(actual, reference)
            for actual, reference in zip(views, expected_views)
        ),
        "views_alias_packed_storage": all(
            value.untyped_storage().data_ptr() == packed_pointer for value in views
        ),
        "last_dim_contiguous": all(value.stride(-1) == 1 for value in views),
        "token_strides": [value.stride(0) for value in views],
        "storage_offsets_elements": [
            value.storage_offset() for value in views
        ],
        "expected_storage_offsets_elements": [0, q_size, q_size + kv_size],
        "row_order_matches_q_k_v": all(
            torch.allclose(actual, reference, atol=0.05, rtol=0.05)
            for actual, reference in zip(views, expected_views)
        ),
    }
    if case["heads"] == case["kv_heads"]:
        grouped = packed.view(
            case["tokens"], 3, case["heads"], case["head_dim"]
        )
        result["grouped_view_matches_split"] = all(
            torch.equal(grouped[:, index], views[index]) for index in range(3)
        )
    return result


def time_case(case, dtype, iterations=10, warmup=3):
    torch.manual_seed(0)
    device = torch.device("cuda")
    x = torch.randn(case["tokens"], case["hidden"], device=device, dtype=dtype)
    output_rows = (case["heads"] + 2 * case["kv_heads"]) * case["head_dim"]
    weight = torch.randn(output_rows, case["hidden"], device=device, dtype=dtype) * 0.05

    def packed_path():
        return packed_projection(x, weight, case["heads"], case["kv_heads"], case["head_dim"])

    def independent_path():
        return independent_projection(x, weight, case["heads"], case["kv_heads"], case["head_dim"])

    def dense_copy_path():
        _, views = packed_path()
        return tuple(value.contiguous() for value in views)

    timings = {}
    for name, function in (
        ("packed_projection_split_views", packed_path),
        ("independent_projections", independent_path),
        ("packed_projection_dense_copies", dense_copy_path),
    ):
        for _ in range(warmup):
            function()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            function()
        end.record()
        torch.cuda.synchronize()
        timings[name] = start.elapsed_time(end) / iterations
    return timings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    started = time.perf_counter()
    import sglang
    import sgl_kernel

    parity = [check_case(case, torch.bfloat16) for case in CASES]
    parity.extend(check_case(case, torch.float16) for case in CASES[-2:])
    timing_case = CASES[-1]
    timings = {
        str(dtype).removeprefix("torch."): time_case(timing_case, dtype)
        for dtype in (torch.bfloat16, torch.float16)
    }
    report = {
        "label": "mirror checkout packed QKV projection representation parity",
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "architecture": ".".join(map(str, torch.cuda.get_device_capability(0))),
        },
        "python": os.path.realpath("/opt/venv/bin/python"),
        "invoked_python": "/opt/venv/bin/python",
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "sglang_import_path": os.path.dirname(sglang.__file__),
        "sgl_kernel_path": os.path.dirname(sgl_kernel.__file__),
        "reference": "three independent torch.nn.functional.linear projections from packed weight rows",
        "timing_method": "CUDA events, 3 warmups, mean of 10 iterations",
        "timing_case": timing_case,
        "first_gpu_execution_elapsed_seconds": time.perf_counter() - started,
        "parity": parity,
        "timings_ms": timings,
    }
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        with open(args.output, "w", encoding="utf-8") as output:
            output.write(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
