from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import sys
import subprocess
from pathlib import Path
from typing import Any

import sglang
import triton
import torch
from torch.profiler import ProfilerActivity, profile


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TEST_PATH = (
    REPOSITORY_ROOT
    / "test"
    / "registered"
    / "kernel"
    / "embeddings"
    / "test_qwen4_ple_offload.py"
)


def load_test_module():
    spec = importlib.util.spec_from_file_location("qwen4_ple_offload_test", TEST_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load test module from {TEST_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_input_batches(token_count: int) -> tuple[torch.Tensor, torch.Tensor]:
    first = torch.arange(token_count * 3, dtype=torch.int64, device="cuda")
    first = (first % 11).reshape(token_count, 3)
    second = torch.arange(3, token_count * 3 + 3, dtype=torch.int64, device="cuda")
    second = (second % 11).reshape(token_count, 3)
    second[:, 0] = 8
    second[:, 2] = -1
    return first, second


def make_reference(
    input_ids: torch.Tensor,
    reference_rows: torch.Tensor,
    embedding_dim: int,
) -> torch.Tensor:
    expected = torch.zeros(
        (*input_ids.shape, embedding_dim), dtype=torch.bfloat16, device="cpu"
    )
    for token_index, token_ids in enumerate(input_ids.cpu().tolist()):
        for head_index, global_id in enumerate(token_ids):
            if 0 <= global_id < reference_rows.shape[0]:
                expected[token_index, head_index] = reference_rows[global_id]
    return expected


def timed_call(
    offloaded: Any,
    input_ids: torch.Tensor,
    output: torch.Tensor | None,
) -> tuple[float, torch.Tensor]:
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    if output is None:
        actual = offloaded.gather(input_ids)
    else:
        actual = offloaded.gather(input_ids, out=output)
    end_event.record()
    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event), actual


def capture_native_dispatch(offloaded: Any, input_ids: torch.Tensor) -> list[str]:
    output = torch.empty(
        (*input_ids.shape, offloaded.embedding_dim),
        dtype=torch.bfloat16,
        device="cuda",
    )
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as result:
        offloaded.gather(input_ids, out=output)
        torch.cuda.synchronize()
    dispatches = []
    for event in result.key_averages():
        if event.device_type != torch.autograd.DeviceType.CPU and event.count > 0:
            dispatches.append(event.key)
    return dispatches


def native_artifact_paths() -> list[str]:
    cache_root = Path(os.environ.get("TRITON_CACHE_DIR", "/tmp/triton"))
    return sorted(
        str(path)
        for path in cache_root.rglob("*gather_ple_embedding_from_pinned_kernel*.hsaco")
    )


def unsupported_results(
    offloaded: Any, input_ids: torch.Tensor
) -> list[dict[str, Any]]:
    embedding_dim = offloaded.embedding_dim
    invalid_outputs = {
        "wrong_shape": torch.empty(
            (*input_ids.shape, embedding_dim + 1), device="cuda"
        ),
        "wrong_dtype": torch.empty(
            (*input_ids.shape, embedding_dim), dtype=torch.float16, device="cuda"
        ),
        "wrong_device": torch.empty(
            (*input_ids.shape, embedding_dim), dtype=torch.bfloat16, device="cpu"
        ),
    }
    records = []
    for case_name, invalid_output in invalid_outputs.items():
        try:
            offloaded.gather(input_ids, out=invalid_output)
        except ValueError as error:
            records.append({"case": case_name, "rejected": True, "error": str(error)})
        else:
            records.append({"case": case_name, "rejected": False, "error": None})
    return records


def run_case(
    test_module: Any,
    storage_dtype: torch.dtype,
    embedding_dim: int,
    token_count: int,
) -> dict[str, Any]:
    source = test_module._make_source_embedding(
        dtype=storage_dtype, embedding_dim=embedding_dim
    )
    offloaded = test_module.Qwen4ExpPinnedHostEmbedding(source)
    source_rows = torch.arange(
        8 * embedding_dim, dtype=torch.bfloat16, device="cuda"
    ).reshape(8, embedding_dim)
    source_rows = source_rows % 15
    test_module._load_rows(offloaded, source_rows.to(dtype=storage_dtype))
    reference_rows = source_rows.to(dtype=storage_dtype).to(
        dtype=torch.bfloat16
    ).cpu()
    input_batches = make_input_batches(token_count)
    expected_batches = [
        make_reference(input_ids, reference_rows, embedding_dim)
        for input_ids in input_batches
    ]

    correctness = []
    for input_ids, expected in zip(input_batches, expected_batches):
        fresh_output = offloaded.gather(input_ids)
        fresh_mismatches = (
            fresh_output.cpu() != expected
        ).sum().item()
        reuse_output = torch.full(
            (*input_ids.shape, embedding_dim),
            torch.nan,
            dtype=torch.bfloat16,
            device="cuda",
        )
        reuse_pointer = reuse_output.data_ptr()
        reused_output = offloaded.gather(input_ids, out=reuse_output)
        reuse_mismatches = (
            reused_output.cpu() != expected
        ).sum().item()
        correctness.append(
            {
                "input_ids_sha256": hashlib.sha256(
                    input_ids.detach().cpu().numpy().tobytes()
                ).hexdigest(),
                "fresh_output_pointer": fresh_output.data_ptr(),
                "fresh_alias_input": fresh_output.data_ptr()
                == input_ids.data_ptr(),
                "fresh_mismatch_count": fresh_mismatches,
                "fresh_max_abs_error": (
                    fresh_output.cpu().float() - expected.float()
                ).abs().max().item(),
                "reuse_output_pointer": reuse_pointer,
                "reuse_return_pointer": reused_output.data_ptr(),
                "reuse_alias_input": reused_output.data_ptr()
                == input_ids.data_ptr(),
                "reuse_sentinel_remaining": bool(
                    torch.isnan(reused_output).any().item()
                ),
                "reuse_mismatch_count": reuse_mismatches,
                "reuse_max_abs_error": (
                    reused_output.cpu().float() - expected.float()
                ).abs().max().item(),
            }
        )

    timings = {"fresh_output_ms": [], "reused_output_ms": []}
    for storage_mode in ("fresh_output_ms", "reused_output_ms"):
        for warmup_index in range(2):
            input_ids = input_batches[warmup_index % len(input_batches)]
            if storage_mode == "fresh_output_ms":
                offloaded.gather(input_ids)
            else:
                reuse_output = torch.empty(
                    (*input_ids.shape, embedding_dim),
                    dtype=torch.bfloat16,
                    device="cuda",
                )
                offloaded.gather(input_ids, out=reuse_output)
        for timed_index in range(10):
            input_ids = input_batches[timed_index % len(input_batches)]
            if storage_mode == "fresh_output_ms":
                elapsed_ms, _ = timed_call(offloaded, input_ids, None)
            else:
                reuse_output = torch.full(
                    (*input_ids.shape, embedding_dim),
                    torch.nan,
                    dtype=torch.bfloat16,
                    device="cuda",
                )
                elapsed_ms, _ = timed_call(offloaded, input_ids, reuse_output)
            timings[storage_mode].append(elapsed_ms)

    dispatches = capture_native_dispatch(offloaded, input_batches[0])
    unsupported = unsupported_results(offloaded, input_batches[0])
    return {
        "case": f"{str(storage_dtype).split('.')[-1]}_dim{embedding_dim}_tokens{token_count}",
        "storage_dtype": str(storage_dtype),
        "output_dtype": "torch.bfloat16",
        "embedding_dim": embedding_dim,
        "token_count": token_count,
        "gathered_rows_per_call": input_batches[0].numel(),
        "distinct_input_batches": 2,
        "reference": "independent CPU row indexing with out-of-range rows zeroed",
        "correctness": correctness,
        "timing": {
            "warmup_calls_per_mode": 2,
            "timed_calls_per_mode": 10,
            "input_batches_alternated": True,
            "fresh_output_ms": timings["fresh_output_ms"],
            "fresh_output_mean_ms": sum(timings["fresh_output_ms"])
            / len(timings["fresh_output_ms"]),
            "reused_output_ms": timings["reused_output_ms"],
            "reused_output_mean_ms": sum(timings["reused_output_ms"])
            / len(timings["reused_output_ms"]),
        },
        "native_dispatches": dispatches,
        "unsupported_output_variants": unsupported,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA-compatible GPU is required")
    test_module = load_test_module()
    cases = []
    for storage_dtype in (torch.bfloat16, torch.float8_e4m3fn):
        for embedding_dim in (7, 64, 257):
            for token_count in (1, 16, 64):
                cases.append(run_case(test_module, storage_dtype, embedding_dim, token_count))
    result = {
        "label": "mirror-checkout gfx942 PLE gather reuse validation",
        "source_revision": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "source_paths": {
            "python_kernel": str(
                REPOSITORY_ROOT
                / "python"
                / "sglang"
                / "srt"
                / "models"
                / "qwen4_exp.py"
            ),
            "test": str(TEST_PATH),
        },
        "runtime_paths": {
            "python": sys.executable,
            "torch": torch.__file__,
            "triton": triton.__file__,
            "sglang": sglang.__file__,
        },
        "native_paths": native_artifact_paths(),
        "image": {
            "requested": "amdpilotv2/open-job-mi300:jit-config-readable-35122-260909",
            "local_image_id": "sha256:dfc9419089c338b5712da4841768b38b1ab79f3da41f8c58c3cd4dfcc1147ff1",
            "verification": "Supplied operator value; no Docker/Podman socket is available for independent inspection.",
        },
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "gfx": "gfx942",
        },
        "python": platform.python_version(),
        "torch": torch.__version__,
        "timing_method": "Two warmups then ten timed calls per mode; CUDA events bracket only the gather. Fresh mode includes allocation; reuse mode has its NaN sentinel fill outside the timed region. Two distinct deterministic input batches alternate.",
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
