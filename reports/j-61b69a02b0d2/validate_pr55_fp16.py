import inspect
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import torch


CHECKOUT = os.environ.get(
    "SGLANG_VALIDATION_CHECKOUT", "/job/sglang-validation-817319"
)
PYTHONPATH = os.path.join(CHECKOUT, "python")
sys.path.insert(0, PYTHONPATH)

from sglang.kernels.ops.kvcache.triton_store_cache import (  # noqa: E402
    try_triton_store_cache_fp8,
)
from sglang.srt.mem_cache import memory_pool  # noqa: E402
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool  # noqa: E402


RESULTS = []
GUARD_CALLS = []
REAL_GUARD = memory_pool.try_triton_store_cache_fp8


def wrapped_guard(*args, **kwargs):
    accepted = REAL_GUARD(*args, **kwargs)
    GUARD_CALLS.append(accepted)
    return accepted


memory_pool.try_triton_store_cache_fp8 = wrapped_guard


def build_pool(head_dim=128, v_head_dim=128):
    return MHATokenToKVPool(
        size=31,
        page_size=1,
        dtype=torch.float8_e4m3fnuz,
        head_num=2,
        head_dim=head_dim,
        v_head_dim=v_head_dim,
        layer_num=1,
        device="cuda",
        enable_memory_saver=False,
        enable_alt_stream=False,
    )


def random_cache(pool):
    for cache in (pool.k_buffer[0], pool.v_buffer[0]):
        cache.copy_(
            torch.randint(
                0,
                256,
                cache.shape,
                device=cache.device,
                dtype=torch.uint8,
            )
        )


def scale_value(scale):
    if scale is None:
        return None
    if isinstance(scale, torch.Tensor):
        return float(scale.detach().cpu().item())
    return float(scale)


def quantize_reference(source, scale, fp8_dtype):
    values = source.detach().cpu().to(torch.float32)
    if scale is not None:
        values = torch.div(values, scale)
    info = torch.finfo(fp8_dtype)
    values = torch.clamp(values, min=info.min, max=info.max)
    return values.to(fp8_dtype).view(torch.uint8)


def complete_reference(pool, k, v, loc, k_scale, v_scale, initial_k, initial_v):
    expected_k = initial_k.detach().cpu().clone()
    expected_v = initial_v.detach().cpu().clone()
    loc_cpu = loc.detach().cpu()
    k_scale_value = scale_value(k_scale)
    v_scale_value = scale_value(v_scale)
    for token in range(loc_cpu.numel()):
        slot = int(loc_cpu[token])
        if slot == 0:
            continue
        expected_k[slot] = quantize_reference(
            k[token], k_scale_value, pool.dtype
        )
        expected_v[slot] = quantize_reference(
            v[token], v_scale_value, pool.dtype
        )
    return expected_k, expected_v


def record(name, passed, detail, fused_calls):
    RESULTS.append(
        {
            "case": name,
            "passed": bool(passed),
            "detail": detail,
            "guard_calls": list(fused_calls),
        }
    )
    print(f"{'PASS' if passed else 'FAIL'} {name}: {detail}")


def bitwise_equal(left, right):
    if left.dtype == torch.bfloat16:
        return torch.equal(
            left.view(torch.int16), right.view(torch.int16)
        )
    if left.dtype == torch.float16:
        return torch.equal(
            left.view(torch.int16), right.view(torch.int16)
        )
    return torch.equal(left, right)


def mismatch_details(actual, expected, limit=16):
    mismatched = actual != expected
    indices = mismatched.nonzero()
    details = []
    for index in indices[:limit]:
        position = tuple(int(value) for value in index)
        details.append(
            {
                "index": position,
                "actual": int(actual[position]),
                "expected": int(expected[position]),
            }
        )
    return details


def run_case(
    name,
    pool,
    k,
    v,
    loc,
    k_scale=None,
    v_scale=None,
    expected_guard=None,
):
    random_cache(pool)
    initial_k = pool.k_buffer[0].detach().clone()
    initial_v = pool.v_buffer[0].detach().clone()
    k_before = k.detach().clone()
    v_before = v.detach().clone()
    GUARD_CALLS.clear()
    pool.set_kv_buffer(
        SimpleNamespace(layer_id=0),
        loc,
        k,
        v,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    torch.cuda.synchronize()
    guard_calls = list(GUARD_CALLS)
    expected_k, expected_v = complete_reference(
        pool,
        k,
        v,
        loc,
        k_scale,
        v_scale,
        initial_k,
        initial_v,
    )
    actual_k = pool.k_buffer[0].detach().cpu()
    actual_v = pool.v_buffer[0].detach().cpu()
    k_equal = torch.equal(actual_k, expected_k)
    v_equal = torch.equal(actual_v, expected_v)
    input_unchanged = bitwise_equal(
        k.detach().cpu(), k_before.cpu()
    ) and bitwise_equal(v.detach().cpu(), v_before.cpu())
    guard_ok = (
        expected_guard is None
        or (expected_guard and any(guard_calls))
        or (not expected_guard and (not guard_calls or not any(guard_calls)))
    )
    mismatched_k = int((actual_k != expected_k).sum())
    mismatched_v = int((actual_v != expected_v).sum())
    passed = k_equal and v_equal and input_unchanged and guard_ok
    record(
        name,
        passed,
        {
            "cache_k_equal": bool(k_equal),
            "cache_v_equal": bool(v_equal),
            "mismatched_k_bytes": mismatched_k,
            "mismatched_v_bytes": mismatched_v,
            "input_unchanged": bool(input_unchanged),
            "guard_selection_ok": bool(guard_ok),
            "first_k_mismatches": mismatch_details(actual_k, expected_k),
            "first_v_mismatches": mismatch_details(actual_v, expected_v),
        },
        guard_calls,
    )
    return passed


def graph_replay_case():
    pool = build_pool()
    loc = torch.tensor([1, 2], device="cuda")
    k = torch.randn(2, 2, 128, device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)
    k_scale = torch.tensor(1.0, device="cuda", dtype=torch.float32)
    v_scale = torch.tensor(1.0, device="cuda", dtype=torch.float32)
    layer = SimpleNamespace(layer_id=0)

    pool.set_kv_buffer(layer, loc, k, v, k_scale, v_scale)
    torch.cuda.synchronize()
    GUARD_CALLS.clear()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        pool.set_kv_buffer(layer, loc, k, v, k_scale, v_scale)
    capture_guard_calls = list(GUARD_CALLS)

    random_cache(pool)
    initial_k = pool.k_buffer[0].detach().clone()
    initial_v = pool.v_buffer[0].detach().clone()
    k_before = k.detach().clone()
    v_before = v.detach().clone()
    loc.copy_(torch.tensor([3, 5], device="cuda"))
    k_scale.fill_(1.25)
    v_scale.fill_(1.75)
    graph.replay()
    torch.cuda.synchronize()
    expected_k, expected_v = complete_reference(
        pool,
        k,
        v,
        loc,
        k_scale,
        v_scale,
        initial_k,
        initial_v,
    )
    actual_k = pool.k_buffer[0].detach().cpu()
    actual_v = pool.v_buffer[0].detach().cpu()
    k_equal = torch.equal(actual_k, expected_k)
    v_equal = torch.equal(actual_v, expected_v)
    input_unchanged = bitwise_equal(
        k.detach().cpu(), k_before.cpu()
    ) and bitwise_equal(v.detach().cpu(), v_before.cpu())
    capture_ok = any(capture_guard_calls)
    passed = k_equal and v_equal and input_unchanged and capture_ok
    record(
        "fp16_cuda_graph_changed_slots_and_device_scales",
        passed,
        {
            "cache_k_equal": bool(k_equal),
            "cache_v_equal": bool(v_equal),
            "mismatched_k_bytes": int((actual_k != expected_k).sum()),
            "mismatched_v_bytes": int((actual_v != expected_v).sum()),
            "input_unchanged": bool(input_unchanged),
            "capture_used_fused_guard": bool(capture_ok),
            "first_k_mismatches": mismatch_details(actual_k, expected_k),
            "first_v_mismatches": mismatch_details(actual_v, expected_v),
        },
        capture_guard_calls,
    )
    return passed


def main():
    torch.manual_seed(55)
    assert torch.cuda.device_count() == 1
    assert torch.cuda.get_device_name(0) == "AMD Instinct MI300X"

    passed = True
    pool = build_pool()
    loc = torch.tensor([0, 2, 5], device="cuda")
    fp16_k = torch.randn(3, 2, 128, device="cuda", dtype=torch.float16)
    fp16_v = torch.randn_like(fp16_k)
    passed &= run_case(
        "fp16_default_scales",
        pool,
        fp16_k,
        fp16_v,
        loc,
        1.0,
        1.0,
        expected_guard=True,
    )
    passed &= run_case(
        "fp16_scalar_scales",
        pool,
        fp16_k,
        fp16_v,
        loc,
        1.25,
        1.75,
        expected_guard=True,
    )
    passed &= run_case(
        "fp16_negative_and_integer_scalar_scales",
        pool,
        fp16_k,
        fp16_v,
        loc,
        -1.25,
        2,
        expected_guard=True,
    )
    passed &= run_case(
        "fp16_device_scales",
        pool,
        fp16_k,
        fp16_v,
        loc,
        torch.tensor(1.25, device="cuda", dtype=torch.float32),
        torch.tensor(1.75, device="cuda", dtype=torch.float32),
        expected_guard=True,
    )
    passed &= run_case(
        "fp16_no_scale_existing_fallback",
        pool,
        fp16_k,
        fp16_v,
        loc,
        expected_guard=False,
    )

    asymmetric_pool = build_pool(head_dim=192, v_head_dim=128)
    asymmetric_k = torch.randn(
        2, 2, 192, device="cuda", dtype=torch.float16
    )
    asymmetric_v = torch.randn(
        2, 2, 128, device="cuda", dtype=torch.float16
    )
    passed &= run_case(
        "fp16_asymmetric_rows",
        asymmetric_pool,
        asymmetric_k,
        asymmetric_v,
        torch.tensor([3, 7], device="cuda"),
        1.5,
        2.5,
        expected_guard=True,
    )

    boundary_pool = build_pool(head_dim=24, v_head_dim=24)
    boundary_values = torch.tensor(
        [
            0.0,
            -0.0,
            0.0009765625,
            -0.0009765625,
            0.001953125,
            -0.001953125,
            0.00390625,
            -0.00390625,
            0.0078125,
            -0.0078125,
            0.015625,
            -0.015625,
            240.0,
            -240.0,
            480.0,
            -480.0,
            496.0,
            -496.0,
            float("inf"),
            float("-inf"),
            float("nan"),
            float("nan"),
            1.0,
            -1.0,
        ],
        device="cuda",
        dtype=torch.float16,
    )
    boundary_k = boundary_values.view(1, 1, 24).expand(2, 2, 24).contiguous()
    passed &= run_case(
        "fp16_saturation_underflow_sign_and_nan_boundaries",
        boundary_pool,
        boundary_k,
        boundary_k,
        torch.tensor([1, 4], device="cuda"),
        2.0,
        2.0,
        expected_guard=True,
    )
    passed &= run_case(
        "fp16_zero_scale_public_guard_admission",
        boundary_pool,
        boundary_k,
        boundary_k,
        torch.tensor([2, 6], device="cuda"),
        0.0,
        0.0,
        expected_guard=True,
    )

    noncontiguous_pool = build_pool()
    k_base = torch.randn(2, 4, 128, device="cuda", dtype=torch.float16)
    v_base = torch.randn_like(k_base)
    noncontiguous_k = k_base[:, ::2, :]
    noncontiguous_v = v_base[:, ::2, :]
    assert not noncontiguous_k.is_contiguous()
    passed &= run_case(
        "fp16_noncontiguous_input_fallback",
        noncontiguous_pool,
        noncontiguous_k,
        noncontiguous_v,
        torch.tensor([2, 4], device="cuda"),
        1.25,
        1.75,
        expected_guard=False,
    )

    unsupported_dtype_scale = torch.tensor(
        1.25, device="cuda", dtype=torch.float16
    )
    passed &= run_case(
        "fp16_unsupported_device_scale_dtype_fallback",
        noncontiguous_pool,
        fp16_k,
        fp16_v,
        loc,
        unsupported_dtype_scale,
        1.75,
        expected_guard=False,
    )

    alignment_pool = build_pool(head_dim=7, v_head_dim=7)
    alignment_k = torch.randn(1, 1, 7, device="cuda", dtype=torch.float16)
    alignment_v = torch.randn_like(alignment_k)
    passed &= run_case(
        "fp16_row_alignment_fallback",
        alignment_pool,
        alignment_k,
        alignment_v,
        torch.tensor([2], device="cuda"),
        1.25,
        1.75,
        expected_guard=False,
    )

    bf16_pool = build_pool()
    bf16_k = torch.randn(3, 2, 128, device="cuda", dtype=torch.bfloat16)
    bf16_v = torch.randn_like(bf16_k)
    passed &= run_case(
        "bf16_scalar_scale_control",
        bf16_pool,
        bf16_k,
        bf16_v,
        loc,
        1.25,
        1.75,
        expected_guard=True,
    )
    passed &= run_case(
        "bf16_device_scale_control",
        bf16_pool,
        bf16_k,
        bf16_v,
        loc,
        torch.tensor(1.25, device="cuda", dtype=torch.float32),
        torch.tensor(1.75, device="cuda", dtype=torch.float32),
        expected_guard=True,
    )

    passed &= graph_replay_case()

    manifest = {
        "checkout": CHECKOUT,
        "pythonpath": PYTHONPATH,
        "imported_memory_pool": inspect.getfile(memory_pool),
        "imported_guard": inspect.getfile(try_triton_store_cache_fp8),
        "torch_version": torch.__version__,
        "hip_version": torch.version.hip,
        "triton_version": __import__("triton").__version__,
        "visible_gpus": torch.cuda.device_count(),
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_arch": torch.cuda.get_device_capability(0),
        "results": RESULTS,
        "all_passed": bool(passed),
    }
    output = os.environ.get(
        "SGLANG_VALIDATION_OUTPUT",
        "/job/native-cache/pr55_fp16_validation.json",
    )
    with open(output, "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
