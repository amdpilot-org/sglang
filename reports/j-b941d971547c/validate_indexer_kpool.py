#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time
from typing import Any


REQUIRED_IMAGE_ID = "sha256:dfc9419089c338b5712da4841768b38b1ab79f3da41f8c58c3cd4dfcc1147ff1"
NUM_HEADS = 32
HEAD_DIM = 128
TOP_K = 8


def run_command(command: list[str], timeout: float = 30.0) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def child_environment(source_root: str) -> dict[str, str]:
    environment = os.environ.copy()
    environment["XDG_CACHE_HOME"] = "/job/.cache"
    environment["TRITON_CACHE_DIR"] = "/job/.cache/triton"
    environment["TORCHINDUCTOR_CACHE_DIR"] = "/job/.cache/torch-inductor"
    environment["TORCH_HOME"] = "/job/.cache/torch"
    environment["HF_HOME"] = "/job/.cache/huggingface"
    environment["PYTHONPATH"] = os.path.join(source_root, "python")
    return environment


def tensor_summary(tensor: Any) -> dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "stride": list(tensor.stride()),
    }


def run_child(args: argparse.Namespace) -> dict[str, Any]:
    source_python = pathlib.Path(args.source_root) / "python"
    sys.path.insert(0, str(source_python))

    import inspect

    import sglang
    import torch
    import triton
    from aiter.ops.triton.fp8_mqa_logits import fp8_mqa_logits
    from sglang.srt.layers.attention.dsa.dsa_indexer_kpool import IndexerKPool

    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    device_properties = torch.cuda.get_device_properties(0)
    free_before, total_memory = torch.cuda.mem_get_info(device)
    torch.cuda.reset_peak_memory_stats()

    result: dict[str, Any] = {
        "case": args.case,
        "kind": args.kind,
        "source_root": args.source_root,
        "source_commit": run_command(["git", "-C", args.source_root, "rev-parse", "HEAD"]),
        "source_status": run_command(["git", "-C", args.source_root, "status", "--short"]),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "sglang_path": sglang.__file__,
        "indexer_kpool_module": IndexerKPool.__module__,
        "indexer_kpool_qualname": IndexerKPool.__qualname__,
        "aiter_wrapper_path": inspect.getmodule(fp8_mqa_logits).__file__,
        "aiter_native_path": "/sgl-workspace/aiter/aiter/jit/module_aiter_core.so",
        "torch_path": torch.__file__,
        "torch_version": torch.__version__,
        "torch_hip_version": torch.version.hip,
        "triton_path": triton.__file__,
        "triton_version": triton.__version__,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_capability": list(torch.cuda.get_device_capability(0)),
        "gpu_total_bytes": device_properties.total_memory,
        "gpu_free_before_bytes": free_before,
        "gpu_total_reported_bytes": total_memory,
        "num_heads": NUM_HEADS,
        "head_dim": HEAD_DIM,
        "top_k": TOP_K,
    }

    if args.kind == "main_support":
        try:
            import deep_gemm

            deep_gemm_status = {
                "available": True,
                "path": getattr(deep_gemm, "__file__", None),
            }
        except Exception as error:
            deep_gemm_status = {"available": False, "error": repr(error)}

        result.update(
            {
                "has_fp8_mqa_logits_helper": hasattr(
                    IndexerKPool, "_fp8_mqa_logits"
                ),
                "has_should_chunk_mqa_logits": hasattr(
                    IndexerKPool, "_should_chunk_mqa_logits"
                ),
                "deep_gemm_status": deep_gemm_status,
            }
        )
        result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        result["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        return result

    num_q = args.num_q
    num_k = args.num_k
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)

    q_bfloat16 = torch.randn(
        (num_q, NUM_HEADS, HEAD_DIM),
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    q_fp8 = q_bfloat16.to(torch.float8_e4m3fn).contiguous()
    del q_bfloat16

    kv_bfloat16 = torch.randn(
        (num_k, HEAD_DIM),
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    k_fp8 = kv_bfloat16.to(torch.float8_e4m3fn).contiguous()
    del kv_bfloat16

    k_scale = torch.ones((num_k,), device=device, dtype=torch.float32)
    weights = torch.randn(
        (num_q, NUM_HEADS),
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    starts = torch.zeros((num_q,), device=device, dtype=torch.int32)
    ends = torch.full((num_q,), num_k, device=device, dtype=torch.int32)

    aligned_num_k = ((num_k + 255) // 256) * 256
    nominal_logits_bytes = num_q * num_k * 4
    allocated_logits_bytes = num_q * aligned_num_k * 4
    result.update(
        {
            "num_q": num_q,
            "num_k": num_k,
            "aligned_num_k": aligned_num_k,
            "nominal_logits_bytes": nominal_logits_bytes,
            "aiter_allocated_logits_bytes": allocated_logits_bytes,
            "nominal_logits_gib": nominal_logits_bytes / (1024**3),
            "aiter_allocated_logits_gib": allocated_logits_bytes / (1024**3),
            "q_fp8": tensor_summary(q_fp8),
            "k_fp8": tensor_summary(k_fp8),
            "weights": tensor_summary(weights),
            "starts": tensor_summary(starts),
            "ends": tensor_summary(ends),
            "seed": args.seed,
        }
    )

    if args.kind == "exactness":
        import sglang.srt.layers.attention.dsa.dsa_indexer_kpool as kpool_module

        expected_rows = args.forced_rows
        kpool_module._MQA_LOGITS_MAX_BYTES_ROCM = expected_rows * num_k * 4
        direct_logits = fp8_mqa_logits(
            q_fp8,
            k_fp8,
            k_scale,
            weights,
            starts,
            ends,
            clean_logits=True,
        )
        torch.cuda.synchronize()
        direct_values, direct_indices = torch.topk(direct_logits, TOP_K, dim=1)
        direct_checksum = torch.sum(direct_logits, dtype=torch.float64).item()
        del direct_logits

        chunked_logits = IndexerKPool._fp8_mqa_logits(
            q_fp8,
            k_fp8,
            k_scale,
            weights,
            starts,
            ends,
            clean_logits=True,
        )
        torch.cuda.synchronize()
        chunked_values, chunked_indices = torch.topk(chunked_logits, TOP_K, dim=1)
        chunked_checksum = torch.sum(chunked_logits, dtype=torch.float64).item()
        result.update(
            {
                "forced_rows": expected_rows,
                "expected_chunk_count": (num_q + expected_rows - 1) // expected_rows,
                "direct_checksum": direct_checksum,
                "chunked_checksum": chunked_checksum,
                "checksum_equal": direct_checksum == chunked_checksum,
                "topk_values_equal": torch.equal(chunked_values, direct_values),
                "topk_indices_equal": torch.equal(chunked_indices, direct_indices),
                "topk_max_abs_diff": (
                    (chunked_values - direct_values).abs().max().item()
                ),
                "chunked_shape": list(chunked_logits.shape),
            }
        )
    elif args.kind == "boundary":
        import sglang.srt.layers.attention.dsa.dsa_indexer_kpool as kpool_module

        natural_max_bytes = getattr(kpool_module, "_MQA_LOGITS_MAX_BYTES_ROCM", None)
        natural_max_rows = (
            natural_max_bytes // (num_k * 4)
            if natural_max_bytes is not None and num_k > 0
            else None
        )
        natural_chunk_rows = None
        if natural_max_rows is not None and 0 < natural_max_rows < num_q:
            natural_chunk_rows = [
                stop - start
                for start, stop in zip(
                    range(0, num_q, natural_max_rows),
                    list(range(natural_max_rows, num_q, natural_max_rows)) + [num_q],
                )
            ]
        if args.forced_rows is not None:
            kpool_module._MQA_LOGITS_MAX_BYTES_ROCM = args.forced_rows * num_k * 4

        logits = IndexerKPool._fp8_mqa_logits(
            q_fp8,
            k_fp8,
            k_scale,
            weights,
            starts,
            ends,
            clean_logits=True,
        )
        torch.cuda.synchronize()
        candidate_values, candidate_indices = torch.topk(logits, TOP_K, dim=1)
        candidate_checksum = torch.sum(logits, dtype=torch.float64).item()
        candidate_shape = list(logits.shape)
        del logits

        reference_rows = args.reference_rows
        reference_chunks = []
        for start in range(0, num_q, reference_rows):
            stop = min(start + reference_rows, num_q)
            reference_chunks.append(
                fp8_mqa_logits(
                    q_fp8[start:stop],
                    k_fp8,
                    k_scale,
                    weights[start:stop],
                    starts[start:stop],
                    ends[start:stop],
                    clean_logits=True,
                )
            )
        reference_logits = torch.cat(reference_chunks, dim=0)
        del reference_chunks
        torch.cuda.synchronize()
        reference_values, reference_indices = torch.topk(reference_logits, TOP_K, dim=1)
        reference_checksum = torch.sum(reference_logits, dtype=torch.float64).item()
        del reference_logits

        result.update(
            {
                "candidate_shape": candidate_shape,
                "candidate_checksum": candidate_checksum,
                "natural_max_rows": natural_max_rows,
                "natural_chunk_rows": natural_chunk_rows,
                "forced_rows": args.forced_rows,
                "reference_rows": reference_rows,
                "reference_chunk_count": (
                    (num_q + reference_rows - 1) // reference_rows
                ),
                "reference_checksum": reference_checksum,
                "checksum_equal": candidate_checksum == reference_checksum,
                "topk_values_equal": torch.equal(candidate_values, reference_values),
                "topk_indices_equal": torch.equal(candidate_indices, reference_indices),
                "topk_max_abs_diff": (
                    (candidate_values - reference_values).abs().max().item()
                ),
            }
        )
    else:
        raise ValueError(f"unsupported child kind: {args.kind}")

    result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    result["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    result["gpu_free_after_bytes"] = torch.cuda.mem_get_info(device)[0]
    return result


def run_suite(args: argparse.Namespace) -> dict[str, Any]:
    cases = [
        {
            "case": "main_support_probe",
            "kind": "main_support",
            "source_root": args.main_root,
        },
        {
            "case": "candidate_forced_chunk_exactness_8192x20000",
            "kind": "exactness",
            "source_root": args.candidate_root,
            "num_q": 8192,
            "num_k": 20000,
            "forced_rows": 1000,
            "seed": 37478,
        },
        {
            "case": "unpatched_below_boundary_23168x23168",
            "kind": "boundary",
            "source_root": args.unpatched_root,
            "num_q": 23168,
            "num_k": 23168,
            "reference_rows": 4096,
            "seed": 37478,
        },
        {
            "case": "unpatched_above_boundary_23180x23180",
            "kind": "boundary",
            "source_root": args.unpatched_root,
            "num_q": 23180,
            "num_k": 23180,
            "reference_rows": 4096,
            "seed": 37478,
        },
        {
            "case": "unpatched_production_shape_8192x120000",
            "kind": "boundary",
            "source_root": args.unpatched_root,
            "num_q": 8192,
            "num_k": 120000,
            "reference_rows": 4096,
            "seed": 37478,
        },
        {
            "case": "candidate_above_boundary_23180x23180",
            "kind": "boundary",
            "source_root": args.candidate_root,
            "num_q": 23180,
            "num_k": 23180,
            "reference_rows": 4096,
            "seed": 37478,
        },
        {
            "case": "candidate_production_shape_8192x120000",
            "kind": "boundary",
            "source_root": args.candidate_root,
            "num_q": 8192,
            "num_k": 120000,
            "reference_rows": 4096,
            "seed": 37478,
        },
        {
            "case": "candidate_balanced_above_boundary_23180x23180",
            "kind": "boundary",
            "source_root": args.candidate_root,
            "num_q": 23180,
            "num_k": 23180,
            "forced_rows": 11590,
            "reference_rows": 4096,
            "seed": 37478,
        },
    ]

    suite_result: dict[str, Any] = {
        "required_image_id": REQUIRED_IMAGE_ID,
        "container_hostname": run_command(["hostname"]).get("stdout"),
        "hostname_is_not_image_identity": True,
        "main_root": args.main_root,
        "unpatched_root": args.unpatched_root,
        "candidate_root": args.candidate_root,
        "python_executable": args.python,
        "case_timeout_seconds": args.timeout,
        "gpu_command": run_command(
            [
                "rocm-smi",
                "--showproductname",
                "--showmeminfo",
                "vram",
            ]
        ),
        "aiter_commit": run_command(
            ["git", "-C", "/sgl-workspace/aiter", "rev-parse", "HEAD"]
        ),
        "aiter_status": run_command(
            ["git", "-C", "/sgl-workspace/aiter", "status", "--short"]
        ),
        "cases": [],
    }

    for case in cases:
        command = [
            args.python,
            str(pathlib.Path(__file__).resolve()),
            "--child",
            "--case",
            case["case"],
            "--kind",
            case["kind"],
            "--source-root",
            case["source_root"],
        ]
        for key in (
            "num_q",
            "num_k",
            "forced_rows",
            "reference_rows",
            "seed",
        ):
            if key in case:
                command.extend([f"--{key.replace('_', '-')}", str(case[key])])

        started = time.monotonic()
        try:
            completed = subprocess.run(
                command,
                env=child_environment(case["source_root"]),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=args.timeout,
                check=False,
            )
            child_result: dict[str, Any] | None = None
            if completed.stdout.strip():
                try:
                    child_result = json.loads(completed.stdout)
                except json.JSONDecodeError as error:
                    child_result = {"json_error": repr(error), "raw_stdout": completed.stdout}
            case_result = {
                "case": case["case"],
                "command": command,
                "returncode": completed.returncode,
                "duration_seconds": time.monotonic() - started,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "result": child_result,
            }
        except subprocess.TimeoutExpired as error:
            case_result = {
                "case": case["case"],
                "command": command,
                "returncode": None,
                "timeout": True,
                "duration_seconds": time.monotonic() - started,
                "stdout": error.stdout,
                "stderr": error.stderr,
                "result": None,
            }
        suite_result["cases"].append(case_result)

    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as output_file:
        json.dump(suite_result, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    return suite_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--case", default="")
    parser.add_argument("--kind", choices=("main_support", "exactness", "boundary"))
    parser.add_argument("--source-root", default=os.getcwd())
    parser.add_argument("--num-q", type=int)
    parser.add_argument("--num-k", type=int)
    parser.add_argument("--forced-rows", type=int)
    parser.add_argument("--reference-rows", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--main-root", default="/job/sglang")
    parser.add_argument("--unpatched-root", default="/job/sglang-unpatched")
    parser.add_argument("--candidate-root", default="/job/sglang-candidate")
    parser.add_argument("--python", default="/opt/venv/bin/python")
    parser.add_argument("--timeout", type=float, default=420.0)
    parser.add_argument("--output", default="/job/investigation-logs/indexer_kpool_results.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.child:
        print(json.dumps(run_child(args), sort_keys=True))
    else:
        run_suite(args)



if __name__ == "__main__":
    main()
