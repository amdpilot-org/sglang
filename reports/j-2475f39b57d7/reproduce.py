import json
import os
import pathlib
import subprocess
import time

import torch

from sglang.kernels.ops.moe import moe_route_radix4


NUM_EXPERTS = 896
TOPK = 16
RTOL = 1e-5
ATOL = 1e-6
SCALING = 2.5

TIE_LANE_RANK = [
    56, 57, 58, 59, 63, 62, 61, 60, 52, 53, 54, 55, 51, 50, 49, 48,
    40, 41, 42, 43, 47, 46, 45, 44, 36, 37, 38, 39, 35, 34, 33, 32,
    24, 25, 26, 27, 31, 30, 29, 28, 20, 21, 22, 23, 19, 18, 17, 16,
    8, 9, 10, 11, 15, 14, 13, 12, 4, 5, 6, 7, 3, 2, 1, 0,
]


def tie_priority(expert):
    group = expert >> 2
    lane = group & 63
    bank = group >> 6
    rank = TIE_LANE_RANK[lane]
    group_rank = rank * 3 + bank if rank < 32 else 96 + (rank - 32) * 4 + bank
    return (group_rank << 2) + (expert & 3)


TIE_PRIORITY = torch.tensor(
    [tie_priority(expert) for expert in range(NUM_EXPERTS)], device="cuda"
)


def distinct_bf16_scores(num_rows):
    bit_patterns = torch.arange(0, 0x8000, dtype=torch.int16)
    all_values = bit_patterns.view(torch.bfloat16).float()
    finite = (
        torch.isfinite(all_values)
        & (all_values.abs() >= 0.001)
        & (all_values.abs() <= 4.0)
    )
    unique_values = torch.unique(all_values[finite])
    assert unique_values.numel() >= NUM_EXPERTS
    indices = torch.linspace(0, unique_values.numel() - 1, NUM_EXPERTS).round().long()
    values = unique_values[indices]
    assert torch.unique(values).numel() == NUM_EXPERTS
    rows = torch.stack([torch.roll(values, shifts=row) for row in range(num_rows)])
    return rows.to(torch.bfloat16).cuda().contiguous()


def distinct_fp32_scores(num_rows):
    values = torch.linspace(-8.0, 8.0, NUM_EXPERTS, dtype=torch.float32)
    rows = torch.stack([torch.roll(values, shifts=row) for row in range(num_rows)])
    return rows.cuda().contiguous()


def torch_reference(scores, bias, renormalize):
    sigmoid = torch.sigmoid(scores.float())
    keys = sigmoid + bias.float()
    cpu_keys = keys.cpu()
    cpu_priority = TIE_PRIORITY.cpu().tolist()
    orders = []
    for row in range(scores.shape[0]):
        row_keys = cpu_keys[row].tolist()
        orders.append(
            sorted(
                range(NUM_EXPERTS),
                key=lambda expert: (-row_keys[expert], cpu_priority[expert]),
            )[:TOPK]
        )
    ids = torch.tensor(orders, dtype=torch.int32, device="cuda")
    weights = torch.gather(sigmoid, 1, ids.long())
    if renormalize:
        total = weights.sum(-1, keepdim=True)
        weights = weights / torch.where(total > 0, total, torch.ones_like(total))
    return weights * SCALING, ids


def aiter_reference(scores, bias, renormalize):
    from aiter import biased_grouped_topk

    weights = torch.empty(
        (scores.shape[0], TOPK), dtype=torch.float32, device="cuda"
    )
    ids = torch.empty((scores.shape[0], TOPK), dtype=torch.int32, device="cuda")
    biased_grouped_topk(scores, bias, weights, ids, 1, 1, renormalize, SCALING)
    return weights, ids


def route(scores, bias, renormalize):
    weights, ids = moe_route_radix4.route_radix4(
        scores, bias, TOPK, renormalize, SCALING
    )
    torch.cuda.synchronize()
    return weights, ids


def run_case(scores, bias, data_perm, label_map, renormalize, use_aiter):
    permuted_scores = scores[:, data_perm].contiguous()
    permuted_bias = bias[data_perm].contiguous()
    base_weights, base_ids = route(scores, bias, renormalize)
    actual_weights, actual_ids = route(
        permuted_scores, permuted_bias, renormalize
    )
    expected_ids = label_map[base_ids.long()]
    reference_weights, reference_ids = (
        aiter_reference(permuted_scores, permuted_bias, renormalize)
        if use_aiter
        else torch_reference(permuted_scores, permuted_bias, renormalize)
    )
    torch.cuda.synchronize()
    return {
        "ids_equal": torch.equal(actual_ids, expected_ids),
        "weights_close": torch.allclose(
            actual_weights, base_weights, rtol=RTOL, atol=ATOL
        ),
        "max_weight_abs_delta": float(
            (actual_weights - base_weights).abs().max()
        ),
        "reference": "aiter" if use_aiter else "pure-torch",
        "reference_ids_equal": torch.equal(actual_ids, reference_ids),
        "reference_weights_close": torch.allclose(
            actual_weights, reference_weights, rtol=RTOL, atol=ATOL
        ),
    }


def run_tie_boundary(data_perm, label_map):
    scores = torch.full((1, NUM_EXPERTS), -5.0, dtype=torch.float32, device="cuda")
    scores[0, :10] = torch.linspace(8.0, 5.75, 10, device="cuda")
    scores[0, 10:40] = 1.0
    bias = torch.zeros(NUM_EXPERTS, dtype=torch.float32, device="cuda")
    base_weights, base_ids = route(scores, bias, True)
    permuted_scores = scores[:, data_perm].contiguous()
    permuted_bias = bias[data_perm].contiguous()
    actual_weights, actual_ids = route(permuted_scores, permuted_bias, True)
    expected_ids = label_map[base_ids.long()]
    aiter_weights, aiter_ids = aiter_reference(
        permuted_scores, permuted_bias, True
    )
    torch_weights, torch_ids = torch_reference(
        permuted_scores, permuted_bias, True
    )
    torch.cuda.synchronize()
    return {
        "input": "1 finite fp32 token; 10 distinct winners, 30 exactly tied candidates, 859 losers",
        "mapped_ids_equal": torch.equal(actual_ids, expected_ids),
        "aiter_ids_equal": torch.equal(actual_ids, aiter_ids),
        "aiter_weights_close": torch.allclose(
            actual_weights, aiter_weights, rtol=RTOL, atol=ATOL
        ),
        "cpu_oracle_ids_equal": torch.equal(actual_ids, torch_ids),
        "cpu_oracle_weights_close": torch.allclose(
            actual_weights, torch_weights, rtol=RTOL, atol=ATOL
        ),
        "base_ids": base_ids[0].tolist(),
        "permuted_ids": actual_ids[0].tolist(),
        "mapped_base_ids": expected_ids[0].tolist(),
    }


def native_module_path():
    cache_root = pathlib.Path(
        os.environ.get("SGLANG_JIT_CACHE_DIR", "~/.cache/sglang/jit")
    ).expanduser()
    candidates = sorted(
        cache_root.rglob("sgl_kernel_jit_moe_route_radix4.so"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return str(candidates[0]) if candidates else None


def main():
    assert torch.cuda.is_available()
    properties = torch.cuda.get_device_properties(0)
    assert "gfx942" in properties.gcnArchName
    assert moe_route_radix4.supported_hardware()
    moe_route_radix4.build()

    generator = torch.Generator().manual_seed(2475)
    permutations = {
        "identity": torch.arange(NUM_EXPERTS),
        "reverse": torch.arange(NUM_EXPERTS - 1, -1, -1),
        "random_1": torch.randperm(NUM_EXPERTS, generator=generator),
        "random_2": torch.randperm(
            NUM_EXPERTS, generator=torch.Generator().manual_seed(97531)
        ),
    }
    permutations = {
        name: permutation.cuda() for name, permutation in permutations.items()
    }
    inverse_permutations = {
        name: torch.empty_like(permutation).scatter_(
            0,
            permutation,
            torch.arange(NUM_EXPERTS, device=permutation.device),
        )
        for name, permutation in permutations.items()
    }

    results = []
    start = time.perf_counter()
    for dtype in (torch.bfloat16, torch.float32):
        for num_tokens in (1, 4, 16):
            scores = (
                distinct_bf16_scores(num_tokens)
                if dtype == torch.bfloat16
                else distinct_fp32_scores(num_tokens)
            )
            bias = torch.zeros(NUM_EXPERTS, dtype=dtype, device="cuda")
            for renormalize in (False, True):
                for name, permutation in permutations.items():
                    result = run_case(
                        scores,
                        bias,
                        permutation,
                        inverse_permutations[name],
                        renormalize,
                        use_aiter=(num_tokens == 1),
                    )
                    result.update(
                        dtype=str(dtype).removeprefix("torch."),
                        num_tokens=num_tokens,
                        renormalize=renormalize,
                        permutation=name,
                    )
                    results.append(result)
    tie_boundary = run_tie_boundary(
        permutations["random_2"], inverse_permutations["random_2"]
    )
    elapsed = time.perf_counter() - start

    import aiter
    import sglang

    repo_root = pathlib.Path(__file__).resolve().parents[2]
    source_head = subprocess.check_output(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True
    ).strip()
    output = {
        "label": "current-mirror gfx942 expert-route permutation investigation",
        "campaign": "repo-e2e-20260909",
        "gpu": {
            "name": properties.name,
            "gcn_arch_name": properties.gcnArchName,
            "uuid": str(properties.uuid),
        },
        "image": {
            "qualified_image": "amdpilotv2/open-job-mi300:jit-config-readable-35122-260909",
            "local_image_id_sha256": "dfc9419089c338b5712da4841768b38b1ab79f3da41f8c58c3cd4dfcc1147ff1",
            "identity_source": "operator-provided local image ID",
        },
        "stack": {
            "python": "/opt/venv/bin/python",
            "torch_version": torch.__version__,
            "torch_path": torch.__file__,
            "sglang_version": sglang.__version__,
            "sglang_path": sglang.__file__,
            "sgl_kernel_path": moe_route_radix4.__file__,
            "aiter_path": aiter.__file__,
        },
        "source": {
            "head": source_head,
            "kernel_python": "/job/sglang/python/sglang/kernels/ops/moe/moe_route_radix4.py",
            "kernel_native_source": "/job/sglang/python/sglang/kernels/jit/csrc/moe/route_radix4_hip.cuh",
            "native_module": native_module_path(),
        },
        "supported_layouts": {
            "num_experts": NUM_EXPERTS,
            "topk": TOPK,
            "dtypes": ["bfloat16", "float32"],
            "token_counts": [1, 4, 16],
            "scores_layout": "row-contiguous [tokens, 896]",
            "bias_layout": "contiguous [896]",
            "grouping": "ungrouped",
        },
        "finite_input_matrix": {
            "tie_free": "896 distinct finite values per row; bf16 values are selected from [-4, 4] and fp32 values from [-8, 8]",
            "tie_boundary": "10 distinct finite winners, 30 exactly tied finite candidates, 859 finite losers",
        },
        "numerical_gates": {
            "ids_exact": True,
            "weights_rtol": RTOL,
            "weights_atol": ATOL,
            "changed": False,
        },
        "timing": {
            "method": "one perf_counter interval around 48 bounded permutation cases plus one tie-boundary case; synchronize after each kernel call; no warmup or occupancy loop",
            "elapsed_seconds": elapsed,
        },
        "results": results,
        "tie_boundary": tie_boundary,
        "summary": {
            "case_count": len(results),
            "supported_property_passed": all(
                result["ids_equal"]
                and result["weights_close"]
                and result["reference_ids_equal"]
                and result["reference_weights_close"]
                for result in results
            ),
            "max_weight_abs_delta": max(
                result["max_weight_abs_delta"] for result in results
            ),
            "tie_boundary_permutation_invariant": tie_boundary["mapped_ids_equal"],
            "tie_boundary_kernel_matches_aiter": tie_boundary["aiter_ids_equal"]
            and tie_boundary["aiter_weights_close"],
            "tie_boundary_kernel_matches_cpu_oracle": tie_boundary[
                "cpu_oracle_ids_equal"
            ]
            and tie_boundary["cpu_oracle_weights_close"],
        },
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
