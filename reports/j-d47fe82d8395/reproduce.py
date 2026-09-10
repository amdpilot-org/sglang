import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--start-ns", type=int, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
script_started = time.perf_counter()

import torch
import triton
import sglang
from sglang.kernels.ops.moe.ep_moe_kernels import ep_gather, ep_scatter

assert torch.cuda.is_available() and torch.cuda.device_count() == 1
device = torch.device("cuda", 0)
torch.cuda.set_device(device)

CASES = [
    {
        "name": "top1_mixed_padded_groups",
        "padded_counts": [128, 256, 128, 128],
        "valid_counts": [127, 1, 128, 0],
        "topk": 1,
        "hidden_size": 128,
        "weights": [1.0],
    },
    {
        "name": "top2_one_hot_inverse",
        "padded_counts": [128] * 8,
        "valid_counts": [0, 1, 127, 128, 0, 0, 0, 0],
        "topk": 2,
        "hidden_size": 128,
        "weights": [1.0, 0.0],
    },
    {
        "name": "top2_half_weight_inverse",
        "padded_counts": [128] * 8,
        "valid_counts": [0, 1, 127, 128, 0, 0, 0, 0],
        "topk": 2,
        "hidden_size": 128,
        "weights": [0.5, 0.5],
    },
    {
        "name": "top1_zero_padded_expert",
        "padded_counts": [0, 128, 128, 0, 128],
        "valid_counts": [0, 128, 0, 0, 127],
        "topk": 1,
        "hidden_size": 128,
        "weights": [1.0],
    },
    {
        "name": "top1_wide_hidden",
        "padded_counts": [128, 128, 128, 128],
        "valid_counts": [128, 0, 1, 127],
        "topk": 1,
        "hidden_size": 1024,
        "weights": [1.0],
    },
]


def make_source(num_tokens, hidden_size, seed):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    base = torch.randn(num_tokens, hidden_size, generator=generator, dtype=torch.float32)
    base = base.to(device=device, dtype=torch.float16)
    special = torch.tensor(
        [
            0.0,
            -0.0,
            torch.finfo(torch.float16).tiny,
            -torch.finfo(torch.float16).tiny,
            1.0,
            -1.0,
            1.2345,
            -2.675,
            torch.finfo(torch.float16).max,
            -torch.finfo(torch.float16).max,
        ],
        dtype=torch.float16,
        device=device,
    )
    offsets = torch.arange(num_tokens * hidden_size, device=device, dtype=torch.int64)
    pattern = special[offsets.reshape(-1) % special.numel()].reshape(num_tokens, hidden_size)
    source = torch.where((offsets.reshape(num_tokens, hidden_size) % 97) == 0, pattern, base)
    assert source.shape == (num_tokens, hidden_size)
    assert torch.isfinite(source.float()).all()
    return source


def run_case(case, seed):
    padded_counts = case["padded_counts"]
    valid_counts = case["valid_counts"]
    topk = case["topk"]
    hidden_size = case["hidden_size"]
    num_experts = len(padded_counts)
    total_assignments = sum(valid_counts)
    assert total_assignments % topk == 0
    num_tokens = total_assignments // topk
    total_padded = sum(padded_counts)
    assert total_padded % 128 == 0
    assert all(valid <= padded for valid, padded in zip(valid_counts, padded_counts))

    expert_ids = torch.cat(
        [
            torch.full((count,), expert, dtype=torch.int32, device=device)
            for expert, count in enumerate(valid_counts)
        ]
    )
    assert expert_ids.numel() == total_assignments
    recv_topk = expert_ids.reshape(num_tokens, topk).contiguous()
    recv_x = make_source(num_tokens, hidden_size, seed)
    weights = torch.tensor(
        [case["weights"]] * num_tokens, dtype=torch.float32, device=device
    )
    assert weights.shape == (num_tokens, topk)

    num_recv_tokens_per_expert = torch.tensor(padded_counts, dtype=torch.int32, device=device)
    num_valid_tokens_per_expert = torch.tensor(valid_counts, dtype=torch.int32, device=device)
    expert_start_loc = torch.full((num_experts,), -12345, dtype=torch.int32, device=device)
    m_indices = torch.full((total_padded,), -12345, dtype=torch.int32, device=device)
    output_tensor = torch.full(
        (total_padded, hidden_size), -12345.0, dtype=torch.float16, device=device
    )
    output_index = torch.full(
        (num_tokens, topk), -12345, dtype=torch.int32, device=device
    )
    gather_out = torch.empty(
        (num_tokens, hidden_size), dtype=torch.float16, device=device
    )

    def run_pair():
        ep_scatter(
            recv_x,
            None,
            recv_topk,
            num_recv_tokens_per_expert,
            num_valid_tokens_per_expert,
            expert_start_loc,
            output_tensor,
            None,
            m_indices,
            output_index,
        )
        ep_gather(output_tensor, recv_topk, weights, output_index, gather_out)

    run_pair()
    torch.cuda.synchronize()

    padded_tensor = torch.tensor(padded_counts, dtype=torch.int32, device=device)
    valid_tensor = torch.tensor(valid_counts, dtype=torch.int32, device=device)
    expected_starts = torch.cumsum(padded_tensor, 0) - padded_tensor
    expected_post_scatter_cursors = expected_starts + valid_tensor
    expected_m_indices = torch.cat(
        [
            torch.cat(
                [
                    torch.full((valid,), expert, dtype=torch.int32, device=device),
                    torch.full((padded - valid,), -1, dtype=torch.int32, device=device),
                ]
            )
            for expert, (valid, padded) in enumerate(zip(valid_counts, padded_counts))
        ]
    )
    selected_source = output_tensor[output_index.reshape(-1)]
    source_per_assignment = (
        recv_x.unsqueeze(1).expand(num_tokens, topk, hidden_size).reshape(-1, hidden_size)
    )
    torch_reference = (
        output_tensor[output_index].float() * weights.unsqueeze(-1).float()
    ).sum(dim=1).to(torch.float16)

    checks = {
        "post_scatter_cursor_matches_start_plus_valid": torch.equal(
            expert_start_loc, expected_post_scatter_cursors
        ),
        "m_indices_matches_independent_reference": torch.equal(
            m_indices, expected_m_indices
        ),
        "output_index_all_assigned": bool((output_index >= 0).all()),
        "output_index_in_bounds": bool(
            (output_index >= 0).all() and (output_index < total_padded).all()
        ),
        "output_index_unique": output_index.unique().numel() == total_assignments,
        "m_indices_at_output_index_matches_expert": torch.equal(
            m_indices[output_index.reshape(-1)], expert_ids
        ),
        "scattered_sources_exact": torch.equal(selected_source, source_per_assignment),
        "padding_rows_untouched": torch.equal(
            output_tensor[expected_m_indices < 0],
            torch.full_like(output_tensor[expected_m_indices < 0], -12345.0),
        ),
        "torch_inverse_reference_exact": torch.equal(torch_reference, recv_x),
        "ep_gather_restores_source_exact": torch.equal(gather_out, recv_x),
        "source_finite": bool(torch.isfinite(recv_x.float()).all()),
        "gather_finite": bool(torch.isfinite(gather_out.float()).all()),
    }

    warmups = 3
    timed = 30
    for _ in range(warmups):
        run_pair()
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(timed):
        run_pair()
    end_event.record()
    torch.cuda.synchronize()
    elapsed_ms = start_event.elapsed_time(end_event)

    return {
        "name": case["name"],
        "num_experts": num_experts,
        "hidden_size": hidden_size,
        "topk": topk,
        "num_tokens": num_tokens,
        "padded_counts": padded_counts,
        "valid_counts": valid_counts,
        "total_padded_tokens": total_padded,
        "weights": case["weights"],
        "dtype": "torch.float16",
        "checks": {name: bool(value) for name, value in checks.items()},
        "all_checks_passed": all(bool(value) for value in checks.values()),
        "timing": {
            "method": "one CUDA event pair around 30 scatter+gather launch pairs after 3 warmups",
            "warmups": warmups,
            "timed_launch_pairs": timed,
            "total_cuda_event_ms": elapsed_ms,
            "mean_cuda_event_ms_per_launch_pair": elapsed_ms / timed,
        },
        "raw": {
            "post_scatter_expert_start_loc": expert_start_loc.detach().cpu().tolist(),
            "expected_post_scatter_expert_start_loc": expected_post_scatter_cursors.detach().cpu().tolist(),
            "gather_mismatch_count": int((gather_out != recv_x).sum().item()),
            "gather_max_abs_error": float(
                (gather_out.float() - recv_x.float()).abs().max().item()
            ),
            "torch_reference_mismatch_count": int(
                (torch_reference != recv_x).sum().item()
            ),
            "m_indices_unique_counts": m_indices.detach().cpu().unique(
                return_counts=True
            )[1].detach().cpu().tolist(),
        },
    }


results = []
for index, case in enumerate(CASES):
    result = run_case(case, 0xD47FE8 + index)
    results.append(result)
    if not result["all_checks_passed"]:
        raise AssertionError(f"case {case['name']} failed: {result['checks']}")

first_gpu_elapsed_since_script_start = time.perf_counter() - script_started
first_gpu_elapsed_since_process_start = (time.time_ns() - args.start_ns) / 1e9
properties = torch.cuda.get_device_properties(0)
record = {
    "label": "mirror checkout at main; real gfx942 scatter followed by inverse gather",
    "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "command": (
        "PYTHONPATH=$PWD/python "
        "TRITON_CACHE_DIR=/tmp/sglang-cache-j-d47fe82d8395 "
        "/opt/venv/bin/python reports/j-d47fe82d8395/reproduce.py "
        "--start-ns <shell-date> --output /tmp/mirror-roundtrip-gfx942-results.json"
    ),
    "source": {
        "repository": "https://github.com/amdpilot-org/sglang.git",
        "checkout": str(Path(__file__).resolve().parents[2]),
        "branch": subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parents[2]), "branch", "--show-current"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip(),
        "commit": subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parents[2]), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip(),
        "python_module": str(
            Path(__file__).resolve().parents[2]
            / "python/sglang/kernels/ops/moe/ep_moe_kernels.py"
        ),
        "imported_sglang_path": sglang.__file__,
    },
    "environment": {
        "python": sys.executable,
        "python_version": sys.version,
        "sglang_version": getattr(sglang, "__version__", None),
        "torch_path": torch.__file__,
        "torch_native_module": "/opt/venv/lib/python3.10/site-packages/torch/_C.cpython-310-x86_64-linux-gnu.so",
        "torch_hip_library": "/opt/venv/lib/python3.10/site-packages/torch/lib/libtorch_hip.so",
        "torch_version": torch.__version__,
        "triton_path": triton.__file__,
        "triton_version": triton.__version__,
        "triton_cache_dir": "/tmp/sglang-cache-j-d47fe82d8395",
        "gpu_name": properties.name,
        "gpu_arch": properties.gcnArchName,
        "gpu_uuid": str(properties.uuid),
        "gpu_serial": "692440003949",
        "gpu_compute_units": properties.multi_processor_count,
        "gpu_count_used": 1,
        "operator_specified_local_image_id": "sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1",
        "platform": platform.platform(),
    },
    "independent_reference": {
        "starts_and_padding": "exclusive torch.cumsum of padded counts plus independently constructed expert-id and -1 padding tensor",
        "inverse_gather": "float32 weighted sum of output_tensor[output_index] followed by cast to float16",
        "comparison": "exact torch.equal for indices and restored float16 values",
    },
    "finite_adversarial_input": [
        "0.0",
        "-0.0",
        "float16 tiny",
        "-float16 tiny",
        "1.0",
        "-1.0",
        "1.2345",
        "-2.675",
        "float16 max",
        "-float16 max",
        "deterministic randn seeds 0xD47FE8..0xD47FE8+4",
    ],
    "first_gpu_execution_elapsed_since_script_start_s": first_gpu_elapsed_since_script_start,
    "first_gpu_execution_elapsed_since_process_start_s": first_gpu_elapsed_since_process_start,
    "cases": results,
    "all_cases_passed": all(result["all_checks_passed"] for result in results),
    "limitations": [
        "Local kernel contract only; not distributed expert-parallel or DeepEP transport proof.",
        "The reported intra-CTA store/load race is latent and was not directly reproduced by these deterministic round trips.",
        "No model weights were downloaded and no environment replacement was performed.",
    ],
}
assert record["all_cases_passed"]
args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
print(json.dumps(record, indent=2, sort_keys=True))
