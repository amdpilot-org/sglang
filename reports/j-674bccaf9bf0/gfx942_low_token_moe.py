import json
import os
import time
from datetime import datetime, timezone

import torch
import triton
from sgl_kernel import moe_align_block_size as aot_moe_align_block_size

from sglang.kernels.ops.moe.moe_align_small_numel import moe_align_small_numel


CASES = [
    {"num_tokens": 1, "topk": 2, "num_experts": 8, "block_size": 128},
    {"num_tokens": 1, "topk": 7, "num_experts": 64, "block_size": 128},
    {"num_tokens": 1, "topk": 7, "num_experts": 257, "block_size": 128},
    {"num_tokens": 1, "topk": 16, "num_experts": 896, "block_size": 128},
    {"num_tokens": 5, "topk": 7, "num_experts": 257, "block_size": 128},
    {"num_tokens": 4, "topk": 16, "num_experts": 896, "block_size": 128},
]

WARMUP_CALLS = 5
TIMED_CALLS = 100


def reference(topk_ids, block_size, num_experts):
    numel = topk_ids.numel()
    bucket = (topk_ids.flatten().to(torch.int64) + 1).cpu()
    counts = torch.bincount(bucket, minlength=num_experts + 1)
    padded = ((counts + block_size - 1) // block_size) * block_size
    offsets = torch.cumsum(padded, 0) - padded
    total = int(padded.sum())
    non_empty = torch.nonzero(padded, as_tuple=False).flatten()
    expert_ids = torch.repeat_interleave(
        non_empty - 1, padded[non_empty] // block_size
    ).to(torch.int32)
    sorted_ids = torch.full((total,), numel, dtype=torch.int32)
    cursor = offsets.clone()
    for pair in range(numel):
        expert_bucket = int(bucket[pair])
        sorted_ids[cursor[expert_bucket]] = pair
        cursor[expert_bucket] += 1
    return sorted_ids, expert_ids, total


def allocate(topk_ids, block_size, num_experts):
    numel = topk_ids.numel()
    if numel < num_experts + 1:
        max_num_tokens_padded = numel * block_size
    else:
        max_num_tokens_padded = numel + (num_experts + 1) * (block_size - 1)
    max_num_blocks = triton.cdiv(max_num_tokens_padded, block_size)
    return (
        torch.empty((max_num_tokens_padded,), dtype=torch.int32, device="cuda"),
        torch.empty((max_num_blocks,), dtype=torch.int32, device="cuda"),
        torch.empty((1,), dtype=torch.int32, device="cuda"),
        torch.empty((num_experts + 2,), dtype=torch.int32, device="cuda"),
    )


def run_aot(topk_ids, block_size, num_experts, buffers):
    sorted_ids, expert_ids, num_post_pad, cumsum = buffers
    aot_moe_align_block_size(
        topk_ids,
        num_experts + 1,
        block_size,
        sorted_ids,
        expert_ids,
        num_post_pad,
        cumsum,
        True,
        False,
    )
    return sorted_ids, expert_ids, num_post_pad


def run_triton(topk_ids, block_size, num_experts, buffers):
    sorted_ids, expert_ids, num_post_pad, _ = buffers
    moe_align_small_numel(
        topk_ids,
        num_experts + 1,
        block_size,
        sorted_ids,
        expert_ids,
        num_post_pad,
    )
    return sorted_ids, expert_ids, num_post_pad


def assert_exact(got, expected, block_size):
    got_sorted, got_experts, got_total = got
    expected_sorted, expected_experts, expected_total = expected
    assert int(got_total.item()) == expected_total
    assert torch.equal(got_experts[: expected_total // block_size].cpu(), expected_experts)
    assert torch.equal(got_sorted[:expected_total].cpu(), expected_sorted)


def assert_blockwise(got, expected, block_size):
    got_sorted, got_experts, got_total = got
    expected_sorted, expected_experts, expected_total = expected
    assert int(got_total.item()) == expected_total
    num_blocks = expected_total // block_size
    assert torch.equal(got_experts[:num_blocks].cpu(), expected_experts[:num_blocks])
    got_blocks = got_sorted[:expected_total].view(num_blocks, block_size).cpu()
    expected_blocks = expected_sorted.view(num_blocks, block_size)
    assert torch.equal(got_blocks.sort(dim=1).values, expected_blocks.sort(dim=1).values)


def timed_call(function):
    for _ in range(WARMUP_CALLS):
        function()
    torch.cuda.synchronize()
    start_ns = time.perf_counter_ns()
    for _ in range(TIMED_CALLS):
        function()
    torch.cuda.synchronize()
    elapsed_ns = time.perf_counter_ns() - start_ns
    return {
        "warmup_calls": WARMUP_CALLS,
        "timed_calls": TIMED_CALLS,
        "elapsed_seconds": elapsed_ns / 1_000_000_000,
        "mean_us_per_call": elapsed_ns / 1_000 / TIMED_CALLS,
    }


def profiler_kernels(function):
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as profiler:
        function()
        torch.cuda.synchronize()
    return sorted(
        {
            event.key
            for event in profiler.events()
            if event.device_type == torch.autograd.DeviceType.CUDA
            and "moe" in event.key.lower()
        }
    )


def main():
    torch.manual_seed(32312)
    torch.cuda.set_device(0)
    results = []
    for case in CASES:
        num_tokens = case["num_tokens"]
        topk = case["topk"]
        num_experts = case["num_experts"]
        block_size = case["block_size"]
        topk_ids = torch.stack(
            [
                torch.randperm(num_experts, device="cuda", dtype=torch.int32)[:topk]
                for _ in range(num_tokens)
            ]
        ).contiguous()
        expected = reference(topk_ids, block_size, num_experts)
        case_result = dict(case)
        for backend, runner, assertion in (
            ("aot", run_aot, assert_blockwise),
            ("triton_pairwise", run_triton, assert_exact),
        ):
            buffers = allocate(topk_ids, block_size, num_experts)
            got = runner(topk_ids, block_size, num_experts, buffers)
            torch.cuda.synchronize()
            assertion(got, expected, block_size)
            timing = timed_call(
                lambda runner=runner, buffers=buffers: runner(
                    topk_ids, block_size, num_experts, buffers
                )
            )
            kernels = profiler_kernels(
                lambda runner=runner, buffers=buffers: runner(
                    topk_ids, block_size, num_experts, buffers
                )
            )
            case_result[backend] = {
                "reference_comparison": "exact" if backend == "triton_pairwise" else "blockwise multiset",
                "passed": True,
                **timing,
                "profiler_kernels": kernels,
            }
        results.append(case_result)
        print(json.dumps(case_result, indent=2))

    report = {
        "label": "gfx942 low-token MoE align controls",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "source_root": os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "git_commit": os.environ.get("SGLANG_TEST_COMMIT", "unknown"),
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "device_count": torch.cuda.device_count(),
        },
        "torch_version": torch.__version__,
        "hip_version": torch.version.hip,
        "timing_method": (
            f"{WARMUP_CALLS} warmups, {TIMED_CALLS} calls, one synchronize after "
            "the timed loop; wall clock around preallocated kernel calls"
        ),
        "cases": results,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
