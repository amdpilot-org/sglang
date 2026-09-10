import argparse
import glob
import importlib.metadata
import json
import os
import subprocess
import time
from pathlib import Path

import sgl_kernel
import torch
from sgl_kernel import deepseek_v4_topk_transform_512

from sglang.kernels.ops.attention.dsv4.topk import (
    _jit_topk_v2_module,
    plan_topk_v2,
    topk_transform_paged_v2,
    topk_transform_ragged_v2,
)


PAGE_SIZE = 64
SENTINEL = -12345
CASES = [
    (4, 4096, 512),
    (8, 16384, 1024),
    (2, 65537, 2048),
]


def make_inputs(batch, seq_len, topk, dtype=torch.float32, seed=37892):
    torch.manual_seed(seed + batch * 100003 + seq_len + topk)
    width = (seq_len + 3) & ~3
    scores = torch.randn(batch, width, dtype=dtype, device="cuda")[:, :seq_len]
    lengths = torch.full((batch,), seq_len, dtype=torch.int32, device="cuda")
    num_pages = (seq_len + PAGE_SIZE - 1) // PAGE_SIZE
    page_table = (
        torch.arange(num_pages, dtype=torch.int32, device="cuda")
        .unsqueeze(0)
        .expand(batch, -1)
        .contiguous()
    )
    output = torch.full((batch, topk), SENTINEL, dtype=torch.int32, device="cuda")
    return scores, lengths, page_table, output


def run_v1(scores, lengths, page_table, output, topk):
    deepseek_v4_topk_transform_512(
        scores, lengths, page_table, output, PAGE_SIZE
    )
    torch.cuda.synchronize()


def run_v2(scores, lengths, page_table, output, metadata):
    topk_transform_paged_v2(
        scores, lengths, page_table, output, PAGE_SIZE, metadata
    )
    torch.cuda.synchronize()


def independent_reference(scores, lengths, topk):
    scores_cpu = scores.detach().cpu().clone()
    lengths_cpu = lengths.detach().cpu().tolist()
    expected = []
    for row, length in enumerate(lengths_cpu):
        expected.extend(
            torch.topk(
                scores_cpu[row, :length], min(topk, length), dim=-1
            ).indices.tolist()
        )
    return expected


def compare_output(output, expected):
    actual = output.detach().cpu().flatten().tolist()
    return {
        "expected_count": len(expected),
        "actual_count": len(actual),
        "sentinel_remaining": actual.count(SENTINEL),
        "sorted_indices_match": sorted(actual) == sorted(expected),
        "max_abs_index_delta": max(
            (
                abs(actual_value - expected_value)
                for actual_value, expected_value in zip(
                    sorted(actual), sorted(expected)
                )
            ),
            default=0,
        ),
    }


def time_variant(variant, scores, lengths, page_table, output, metadata):
    for _ in range(2):
        if variant == "v1":
            run_v1(scores, lengths, page_table, output, output.shape[1])
        else:
            run_v2(scores, lengths, page_table, output, metadata)

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(5)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(5)]
    for start_event, end_event in zip(start_events, end_events):
        start_event.record()
        if variant == "v1":
            run_v1(scores, lengths, page_table, output, output.shape[1])
        else:
            run_v2(scores, lengths, page_table, output, metadata)
        end_event.record()
    torch.cuda.synchronize()
    timings = [
        start_event.elapsed_time(end_event)
        for start_event, end_event in zip(start_events, end_events)
    ]
    return {
        "method": "2 warmups then 5 CUDA-event timed launches; median reported",
        "event_timings_ms": timings,
        "median_ms": sorted(timings)[len(timings) // 2],
    }


def probe_timing_matrix():
    results = []
    first_gpu_elapsed = None
    for batch, seq_len, topk in CASES:
        scores, lengths, page_table, output = make_inputs(
            batch, seq_len, topk
        )
        expected = independent_reference(scores, lengths, topk)
        if topk <= 1024:
            call_started = time.perf_counter()
            run_v1(scores, lengths, page_table, output, topk)
            if first_gpu_elapsed is None:
                first_gpu_elapsed = time.perf_counter() - call_started
            results.append(
                {
                    "case": {
                        "batch": batch,
                        "seq_len": seq_len,
                        "topk": topk,
                    },
                    "variant": "v1_aot",
                    "supported": True,
                    **compare_output(output, expected),
                    **time_variant(
                        "v1", scores, lengths, page_table, output, None
                    ),
                }
            )
        else:
            results.append(
                {
                    "case": {
                        "batch": batch,
                        "seq_len": seq_len,
                        "topk": topk,
                    },
                    "variant": "v1_aot",
                    "supported": False,
                    "error": "documented v1 top-k limit is 1024; topk=2048 not forced",
                }
            )

        output.fill_(SENTINEL)
        metadata = plan_topk_v2(lengths)
        call_started = time.perf_counter()
        run_v2(scores, lengths, page_table, output, metadata)
        if first_gpu_elapsed is None:
            first_gpu_elapsed = time.perf_counter() - call_started
        results.append(
            {
                "case": {
                    "batch": batch,
                    "seq_len": seq_len,
                    "topk": topk,
                },
                "variant": "v2_jit",
                "supported": True,
                **compare_output(output, expected),
                **time_variant(
                    "v2", scores, lengths, page_table, output, metadata
                ),
            }
        )
        del scores, lengths, page_table, output, metadata
        torch.cuda.empty_cache()
    return results, first_gpu_elapsed


def probe_dtypes():
    results = []
    for dtype in (torch.float16, torch.bfloat16):
        scores, lengths, page_table, output = make_inputs(
            2, 4096, 512, dtype=dtype
        )
        scores_cpu = scores.detach().cpu().clone()
        reference = torch.topk(scores_cpu, 512, dim=-1).indices
        metadata = plan_topk_v2(lengths)
        for variant in ("v1_aot", "v2_jit"):
            output.fill_(SENTINEL)
            try:
                if variant == "v1_aot":
                    run_v1(scores, lengths, page_table, output, 512)
                else:
                    run_v2(scores, lengths, page_table, output, metadata)
                supported = True
                error = None
            except RuntimeError as exception:
                supported = False
                error = f"{type(exception).__name__}: {exception}"
            results.append(
                {
                    "variant": variant,
                    "dtype": str(dtype),
                    "supported": supported,
                    "error": error,
                    "independent_reference": "CPU torch.topk on a cloned dtype tensor",
                    "reference_count": int(reference.numel()),
                    "sentinel_remaining": int(
                        torch.sum(output == SENTINEL).item()
                    ),
                }
            )
        del scores, lengths, page_table, output, metadata
    return results


def probe_graph_replay():
    results = []
    for variant in ("v1_aot", "v2_jit"):
        scores, lengths, page_table, output = make_inputs(4, 4096, 512)
        metadata = plan_topk_v2(lengths) if variant == "v2_jit" else None
        score_address = scores.data_ptr()
        output_address = output.data_ptr()
        metadata_address = (
            metadata.data_ptr() if metadata is not None else None
        )
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            if variant == "v1_aot":
                deepseek_v4_topk_transform_512(
                    scores, lengths, page_table, output, PAGE_SIZE
                )
            else:
                topk_transform_paged_v2(
                    scores, lengths, None, output, PAGE_SIZE, metadata
                )

        torch.manual_seed(224)
        scores.copy_(
            torch.randn(
                scores.shape, dtype=torch.float32, device=scores.device
            )
        )
        graph.replay()
        torch.cuda.synchronize()
        expected = independent_reference(scores, lengths, 512)
        results.append(
            {
                "variant": variant,
                "static_score_address": score_address,
                "static_output_address": output_address,
                "static_metadata_address": metadata_address,
                "score_address_preserved": scores.data_ptr() == score_address,
                "output_address_preserved": output.data_ptr()
                == output_address,
                **compare_output(output, expected),
            }
        )
        del scores, lengths, page_table, output, metadata
    return results


def probe_ragged_aliasing():
    starts = torch.tensor([1, 4097], dtype=torch.int32, device="cuda")
    lengths = torch.tensor([4096, 8], dtype=torch.int32, device="cuda")
    width = ((int((starts + lengths).max().item())) + 3) & ~3
    torch.manual_seed(224)
    scores = torch.randn(2, width, dtype=torch.float32, device="cuda")
    before = scores.clone()
    output = torch.full((2, 512), SENTINEL, dtype=torch.int32, device="cuda")
    topk_transform_ragged_v2(
        scores,
        lengths,
        out_offsets=starts,
        out_indices=output,
        row_starts=starts,
    )
    torch.cuda.synchronize()

    expected = []
    for row in range(2):
        start = int(starts[row].item())
        length = int(lengths[row].item())
        expected.extend(
            torch.topk(
                before[row, start : start + length].cpu(),
                min(512, length),
                dim=-1,
            ).indices.tolist()
        )

    actual = []
    for row in range(2):
        offset = int(starts[row].item())
        actual.extend(
            value - offset
            for value in output[row].detach().cpu().tolist()
            if value != -1
        )

    changed = (scores != before).cpu()
    stray_writes = []
    allowed_heads = []
    for row in range(2):
        start = int(starts[row].item())
        length = int(lengths[row].item())
        allowed = torch.zeros(width, dtype=torch.bool)
        if length > 512:
            allowed[start - start % 4 : start] = True
            allowed_heads.append([start - start % 4, start])
        stray_writes.extend(
            (changed[row] & ~allowed).nonzero().flatten().tolist()
        )

    return {
        "contract": "ragged v2 may mask only the <=3 columns ahead of each non-trivial window",
        "allowed_head_columns": allowed_heads,
        "stray_writes": stray_writes,
        "sentinel_remaining": int(torch.sum(output == SENTINEL).item()),
        "sorted_indices_match": sorted(actual) == sorted(expected),
    }


def jit_module_path():
    _jit_topk_v2_module()
    cache_root = Path(
        os.environ.get("SGLANG_JIT_CACHE_DIR", "~/.cache/sglang/jit")
    ).expanduser()
    candidates = glob.glob(
        str(
            cache_root
            / "gfx942"
            / "sgl_kernel_jit_dpsk_v4_topk_v2"
            / "**"
            / "sgl_kernel_jit_dpsk_v4_topk_v2.so"
        ),
        recursive=True,
    )
    return max(candidates, key=os.path.getmtime) if candidates else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default=str(Path(__file__).with_name("gfx942-results.json")),
    )
    args = parser.parse_args()

    started = time.perf_counter()
    timing_results, first_gpu_elapsed = probe_timing_matrix()
    properties = torch.cuda.get_device_properties(0)
    repository_root = Path(__file__).resolve().parents[2]
    payload = {
        "label": "persistent mirror checkout probe",
        "campaign": "repo-e2e-20260909",
        "upstream_issue": "sgl-project/sglang issue 37892",
        "prior_context": "amdpilot-org/sglang issue 224",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repository_root, text=True
        ).strip(),
        "command": f"{os.sys.executable} {Path(__file__).resolve()}",
        "gpu": {
            "name": properties.name,
            "capability": [properties.major, properties.minor],
            "count": torch.cuda.device_count(),
        },
        "image": {
            "operator_specified_local_image_id": "sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1",
        },
        "python": {
            "executable": os.sys.executable,
            "version": os.sys.version,
            "torch": torch.__version__,
            "torch_path": torch.__file__,
            "hip": torch.version.hip,
            "sglang_path": __import__("sglang").__file__,
            "sglang_version": importlib.metadata.version("sglang"),
            "sgl_kernel_path": sgl_kernel.__file__,
            "sgl_kernel_version": sgl_kernel.__version__,
            "native_common_ops": "/opt/venv/lib/python3.10/site-packages/sgl_kernel/common_ops.cpython-310-x86_64-linux-gnu.so",
        },
        "native_dispatch": {
            "v1": "ROCm AOT torch op deepseek_v4_topk_transform_512 in common_ops .so",
            "v2": "checkout JIT module sgl_kernel_jit_dpsk_v4_topk_v2 topk_transform_paged",
            "v2_native_module": jit_module_path(),
        },
        "first_gpu_execution_elapsed_seconds": first_gpu_elapsed,
        "timing_matrix": timing_results,
        "dtype_rejection": probe_dtypes(),
        "graph_replay": probe_graph_replay(),
        "ragged_aliasing": probe_ragged_aliasing(),
        "total_elapsed_seconds": time.perf_counter() - started,
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
