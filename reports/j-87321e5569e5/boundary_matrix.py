import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
import triton
import triton.language as tl

import aiter


QK_HEAD_DIM = 576
V_HEAD_DIM = 512
ATOL = 1.6e-1
RTOL = 1.6e-1
PATTERNS = 4


@triton.jit
def _convert_req_index_to_global_index_kernel(
    kv_indptr,
    kv_indices,
    token_indices_ptr,
    out_ptr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_TOPK_TOKENS: tl.constexpr,
    bt_stride0: tl.constexpr,
    ti_stride0: tl.constexpr,
    ti_stride1: tl.constexpr,
    out_stride0: tl.constexpr,
    out_stride1: tl.constexpr,
    qo_len: tl.constexpr,
):
    token_id = tl.program_id(0)
    tile_id = tl.program_id(1)
    indice_id = tile_id * BLOCK_N + tl.arange(0, BLOCK_N)
    batch_id = token_id // qo_len

    kv_start = tl.load(kv_indptr + batch_id)
    kv_end = tl.load(kv_indptr + batch_id + 1)
    kv_len = kv_end - kv_start

    ti_ptr = token_indices_ptr + token_id * ti_stride0 + indice_id * ti_stride1
    tok = tl.load(ti_ptr)
    is_invalid_tok = tok < 0

    block_id = tok // BLOCK_SIZE
    inblock_off = tok % BLOCK_SIZE
    valid_block = indice_id < kv_len
    base = tl.load(
        kv_indices + kv_start + block_id * bt_stride0,
        mask=valid_block,
        other=0,
    )
    out_val = tl.where(
        is_invalid_tok | (~valid_block),
        -1,
        base * BLOCK_SIZE + inblock_off,
    )

    out_ptr_ij = out_ptr + token_id * out_stride0 + indice_id * out_stride1
    tl.store(out_ptr_ij, out_val)


def triton_convert_req_index_to_global_index(
    kv_indptr,
    kv_indices,
    token_indices,
    qo_len,
    topk,
):
    assert kv_indices.dtype == torch.int32
    assert token_indices.dtype == torch.int32
    assert token_indices.shape[1] == topk
    assert topk % 128 == 0

    kv_indptr_c = kv_indptr.contiguous()
    kv_indices_c = kv_indices.contiguous()
    token_indices_c = token_indices.contiguous()
    out = torch.empty_like(token_indices_c)

    bt_stride0 = kv_indices_c.stride()[0]
    ti_stride0, ti_stride1 = token_indices_c.stride()
    out_stride0, out_stride1 = out.stride()
    grid = (token_indices_c.shape[0], topk // 128)

    _convert_req_index_to_global_index_kernel[grid](
        kv_indptr_c,
        kv_indices_c,
        token_indices_c,
        out,
        1,
        128,
        topk,
        bt_stride0,
        ti_stride0,
        ti_stride1,
        out_stride0,
        out_stride1,
        qo_len,
    )
    return out


def make_case(rows, heads, topk, kv_len, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    base_q = (
        torch.randn(PATTERNS, heads, QK_HEAD_DIM, generator=generator)
        .clamp(-2.0, 2.0)
        .to(device="cuda", dtype=torch.bfloat16)
    )
    base_local_indices = torch.stack(
        [
            torch.randperm(kv_len, generator=generator)[:topk]
            for _ in range(PATTERNS)
        ]
    ).to(torch.int32)
    page_mapping = torch.randperm(kv_len, generator=generator).to(torch.int32)
    kv = (
        torch.randn(kv_len, 1, QK_HEAD_DIM, generator=generator)
        .clamp(-2.0, 2.0)
        .to(device="cuda", dtype=torch.bfloat16)
    )

    pattern_ids = torch.arange(rows, device="cuda") % PATTERNS
    base_local_indices_cuda = base_local_indices.to("cuda")
    q = base_q[pattern_ids].contiguous()
    local_indices = base_local_indices_cuda[pattern_ids].contiguous()
    kv_indices = page_mapping.to("cuda").repeat(rows)
    expected_global_indices = page_mapping[base_local_indices].to("cuda")
    return (
        base_q,
        expected_global_indices,
        kv,
        q,
        local_indices,
        kv_indices,
        pattern_ids,
    )


def independent_reference(base_q, expected_global_indices, kv, scale):
    outputs = []
    for pattern in range(base_q.shape[0]):
        keys = kv[expected_global_indices[pattern]].squeeze(1).float()
        scores = torch.einsum(
            "hd,kd->hk",
            base_q[pattern].float(),
            keys,
        ) * scale
        weights = scores.softmax(dim=-1)
        outputs.append(
            torch.einsum(
                "hk,kd->hd",
                weights,
                keys[:, :V_HEAD_DIM],
            )
        )
    return torch.stack(outputs)


def launch(
    q,
    kv,
    local_indices,
    kv_indices,
    heads,
    topk,
    kv_len,
    scale,
):
    rows = q.shape[0]
    output = torch.full(
        (rows, heads, V_HEAD_DIM),
        float("nan"),
        device="cuda",
        dtype=torch.bfloat16,
    )
    qo_indptr = torch.arange(rows + 1, device="cuda", dtype=torch.int32)
    kv_indptr = torch.arange(
        0,
        rows * kv_len + 1,
        kv_len,
        device="cuda",
        dtype=torch.int32,
    )
    last_page_lens = torch.ones(rows, device="cuda", dtype=torch.int32)

    metadata_sizes = aiter.get_mla_metadata_info_v1(
        rows,
        1,
        heads,
        torch.bfloat16,
        torch.bfloat16,
        is_sparse=True,
        fast_mode=True,
        num_kv_splits=32,
    )
    metadata_buffers = [
        torch.empty(size, dtype=dtype, device="cuda")
        for size, dtype in metadata_sizes
    ]
    (
        work_meta_data,
        work_indptr,
        work_info_set,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
    ) = metadata_buffers

    aiter.get_mla_metadata_v1(
        qo_indptr,
        kv_indptr,
        last_page_lens,
        heads,
        1,
        True,
        work_meta_data,
        work_info_set,
        work_indptr,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
        page_size=1,
        kv_granularity=16,
        max_seqlen_qo=1,
        uni_seqlen_qo=1,
        fast_mode=True,
        max_split_per_batch=32,
        topk=topk,
        dtype_q_nope=torch.bfloat16,
        dtype_kv_nope=torch.bfloat16,
    )

    converted_indices = triton_convert_req_index_to_global_index(
        kv_indptr,
        kv_indices,
        local_indices,
        1,
        topk,
    )
    torch.cuda.synchronize()
    started = time.perf_counter()
    aiter.mla.mla_decode_fwd(
        q,
        kv.view(kv.shape[0], 1, 1, QK_HEAD_DIM),
        output,
        qo_indptr,
        kv_indptr,
        converted_indices.reshape(-1),
        last_page_lens,
        1,
        1,
        1,
        scale,
        num_kv_splits=32,
        work_meta_data=work_meta_data,
        work_indptr=work_indptr,
        work_info_set=work_info_set,
        reduce_indptr=reduce_indptr,
        reduce_final_map=reduce_final_map,
        reduce_partial_map=reduce_partial_map,
    )
    torch.cuda.synchronize()
    return output, time.perf_counter() - started


def compare_to_reference(output, reference, pattern_ids):
    output_float = output.float()
    reference_float = reference[pattern_ids]
    abs_diff = (output_float - reference_float).abs()
    return {
        "max_abs_diff": float(abs_diff.max().item()),
        "mean_abs_diff": float(abs_diff.mean().item()),
        "allclose": bool(
            torch.allclose(
                output_float,
                reference_float,
                atol=ATOL,
                rtol=RTOL,
            )
        ),
        "all_finite": bool(torch.isfinite(output_float).all().item()),
        "sentinel_absent": bool((~torch.isnan(output_float)).all().item()),
    }


def compare_to_full(output, full_output):
    difference = (output.float() - full_output.float()).abs()
    return {
        "bitwise_equal_to_full": bool(torch.equal(output, full_output)),
        "full_vs_split_max_abs_diff": float(difference.max().item()),
        "full_vs_split_mean_abs_diff": float(difference.mean().item()),
        "full_vs_split_different_elements": int(
            difference.count_nonzero().item()
        ),
    }


def run_case(name, rows, heads, topk, kv_len, chunkings, seed):
    (
        base_q,
        expected_global_indices,
        kv,
        q,
        local_indices,
        kv_indices,
        pattern_ids,
    ) = make_case(rows, heads, topk, kv_len, seed)
    scale = QK_HEAD_DIM**-0.5
    reference = independent_reference(
        base_q,
        expected_global_indices,
        kv,
        scale,
    )

    full_output = None
    full_timing = None
    full_error = None
    pending_outputs = []
    if rows <= 65535:
        try:
            full_output, full_timing = launch(
                q,
                kv,
                local_indices,
                kv_indices,
                heads,
                topk,
                kv_len,
                scale,
            )
        except BaseException as error:
            full_error = f"{type(error).__name__}: {error}"

    split_results = []
    for chunks in chunkings:
        chunk_results = []
        start = 0
        for chunk in chunks:
            stop = start + chunk
            try:
                output, elapsed = launch(
                    q[start:stop],
                    kv,
                    local_indices[start:stop],
                    kv_indices[start * kv_len : stop * kv_len],
                    heads,
                    topk,
                    kv_len,
                    scale,
                )
                comparison = compare_to_reference(
                    output,
                    reference,
                    pattern_ids[start:stop],
                )
                if full_output is not None:
                    comparison.update(
                        compare_to_full(output, full_output[start:stop])
                    )
                else:
                    comparison["bitwise_equal_to_full"] = None
                    pending_outputs.append(
                        {
                            "chunking_index": len(split_results),
                            "chunk_index": len(chunk_results),
                            "start": start,
                            "stop": stop,
                            "output": output,
                        }
                    )
                chunk_results.append(
                    {
                        "rows": chunk,
                        "elapsed_seconds": elapsed,
                        **comparison,
                    }
                )
            except BaseException as error:
                chunk_results.append(
                    {
                        "rows": chunk,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
            start = stop
        split_results.append({"chunks": chunks, "results": chunk_results})

    if rows > 65535:
        try:
            full_output, full_timing = launch(
                q,
                kv,
                local_indices,
                kv_indices,
                heads,
                topk,
                kv_len,
                scale,
            )
        except BaseException as error:
            full_error = f"{type(error).__name__}: {error}"

    if full_output is not None:
        for pending in pending_outputs:
            split_results[pending["chunking_index"]]["results"][
                pending["chunk_index"]
            ].update(
                compare_to_full(
                    pending["output"],
                    full_output[pending["start"] : pending["stop"]],
                )
            )
            del pending["output"]

    full_comparison = None
    if full_output is not None:
        full_comparison = compare_to_reference(
            full_output,
            reference,
            pattern_ids,
        )

    result = {
        "name": name,
        "rows": rows,
        "heads": heads,
        "topk": topk,
        "kv_len": kv_len,
        "full_launch": {
            "elapsed_seconds": full_timing,
            "error": full_error,
            **(full_comparison or {}),
        },
        "split_launches": split_results,
    }
    del full_output, q, local_indices, kv_indices, reference
    torch.cuda.empty_cache()
    return result


def unsupported_probe(heads):
    rows, topk, kv_len = 4, 128, 256
    (
        _,
        _,
        kv,
        q,
        local_indices,
        kv_indices,
        _,
    ) = make_case(rows, heads, topk, kv_len, 9)
    try:
        launch(
            q,
            kv,
            local_indices,
            kv_indices,
            heads,
            topk,
            kv_len,
            QK_HEAD_DIM**-0.5,
        )
    except BaseException as error:
        print(f"{type(error).__name__}: {error}")
        return 1
    return 0


def capture_command(command):
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return {
        "command": " ".join(command),
        "exit_status": completed.returncode,
        "output": completed.stdout.strip(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unsupported-heads", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.unsupported_heads is not None:
        raise SystemExit(unsupported_probe(args.unsupported_heads))

    cases = [
        (
            "rows4_topk128",
            4,
            16,
            128,
            2048,
            [[2, 2], [1, 3]],
            101,
        ),
        (
            "rows4_topk256",
            4,
            16,
            256,
            2048,
            [[2, 2], [1, 3]],
            102,
        ),
        (
            "rows65535_topk128",
            65535,
            16,
            128,
            2048,
            [[32768, 32767], [1, 65534]],
            103,
        ),
        (
            "rows65535_topk256",
            65535,
            16,
            256,
            2048,
            [[32768, 32767], [1, 65534]],
            104,
        ),
        (
            "rows65536_topk128",
            65536,
            16,
            128,
            2048,
            [[32768, 32768], [65535, 1], [1, 65535]],
            105,
        ),
        (
            "rows65536_topk256",
            65536,
            16,
            256,
            2048,
            [[32768, 32768], [65535, 1], [1, 65535]],
            106,
        ),
    ]

    results = []
    for name, rows, heads, topk, kv_len, chunkings, seed in cases:
        results.append(
            run_case(name, rows, heads, topk, kv_len, chunkings, seed)
        )

    unsupported = [
        capture_command(
            [
                sys.executable,
                "-c",
                "import flashinfer; print(flashinfer.__file__)",
            ]
        ),
        capture_command(
            [
                sys.executable,
                "-c",
                "from sgl_kernel import flash_mla_ops; print(flash_mla_ops)",
            ]
        ),
        capture_command(
            [
                sys.executable,
                str(Path(__file__)),
                "--unsupported-heads",
                "4",
            ]
        ),
    ]

    report = {
        "label": "persistent-checkout gfx942 boundary matrix",
        "image": {
            "expected_local_image_id": (
                "sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1"
            ),
            "identity_method": "operator-provided local image ID",
        },
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "torch_hip_version": torch.version.hip,
        },
        "paths": {
            "python": sys.executable,
            "torch": torch.__file__,
            "triton": triton.__file__,
            "aiter": aiter.__file__,
            "sglang_checkout": str(Path(__file__).resolve().parents[2]),
        },
        "kernel": "aiter.mla.mla_decode_fwd",
        "operation_contract": (
            "stock sparse MLA path: per-request kv_indptr, persistent page table, "
            "Triton request-index-to-global-index conversion, page_size=1"
        ),
        "reference": (
            "independent FP32 gathered-key softmax and value reduction"
        ),
        "numerical_gate": {"atol": ATOL, "rtol": RTOL},
        "timing_method": (
            "time.perf_counter around one kernel launch with "
            "torch.cuda.synchronize before and after; no warmup or repeats"
        ),
        "cases": results,
        "unsupported": unsupported,
    }
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
