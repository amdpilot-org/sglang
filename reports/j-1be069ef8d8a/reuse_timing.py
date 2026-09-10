import json
import statistics
import sys
import time
from pathlib import Path

import torch

from sglang.kernels.ops.attention.extend_attention import extend_attention_fwd_unified


CONFIGS = [
    {
        "name": "small",
        "prefix_lens": [3],
        "extend_lens": [2],
        "H_Q": 2,
        "H_KV": 1,
        "D": 64,
    },
    {
        "name": "medium",
        "prefix_lens": [3, 5],
        "extend_lens": [4, 2],
        "H_Q": 4,
        "H_KV": 2,
        "D": 64,
    },
    {
        "name": "large",
        "prefix_lens": [3, 5, 2, 4],
        "extend_lens": [4, 2, 3, 1],
        "H_Q": 8,
        "H_KV": 2,
        "D": 128,
    },
]


def cumsum(values):
    return [0, *torch.tensor(values).cumsum(0).tolist()]


def make_case(config, device):
    prefix_lens = config["prefix_lens"]
    extend_lens = config["extend_lens"]
    total_tokens = sum(prefix_lens) + sum(extend_lens)
    extend_tokens = sum(extend_lens)
    H_Q, H_KV, D = config["H_Q"], config["H_KV"], config["D"]

    q = torch.empty((extend_tokens, H_Q, D), dtype=torch.float16, device=device)
    k_buffer = torch.empty((total_tokens, H_KV, D), dtype=torch.bfloat16, device=device)
    v_buffer = torch.empty((total_tokens, H_KV, D), dtype=torch.bfloat16, device=device)
    o_reused = torch.empty_like(q, dtype=torch.bfloat16)

    qo_indptr = torch.tensor(cumsum(extend_lens), dtype=torch.int32, device=device)
    total_lens = [
        prefix_len + extend_len
        for prefix_len, extend_len in zip(prefix_lens, extend_lens)
    ]
    kv_indptr = torch.tensor(cumsum(total_lens), dtype=torch.int32, device=device)
    unified_kv_indices = torch.arange(total_tokens, dtype=torch.int64, device=device)
    prefix_lens_tensor = torch.tensor(prefix_lens, dtype=torch.int32, device=device)
    max_len_extend = max(extend_lens)

    prefix_indices = []
    cursor = 0
    for prefix_len, extend_len in zip(prefix_lens, extend_lens):
        prefix_indices.extend(range(cursor, cursor + prefix_len))
        cursor += prefix_len + extend_len
    prefix_kv_indices = torch.tensor(prefix_indices, dtype=torch.int64, device=device)
    prefix_kv_indptr = torch.tensor(
        [0, *torch.tensor(prefix_lens).cumsum(0).tolist()],
        dtype=torch.int32,
        device=device,
    )

    return {
        "q": q,
        "k_buffer": k_buffer,
        "v_buffer": v_buffer,
        "o_reused": o_reused,
        "qo_indptr": qo_indptr,
        "kv_indptr": kv_indptr,
        "unified_kv_indices": unified_kv_indices,
        "prefix_lens": prefix_lens_tensor,
        "max_len_extend": max_len_extend,
        "prefix_kv_indptr": prefix_kv_indptr,
        "prefix_kv_indices": prefix_kv_indices,
        "prefix_lens_list": prefix_lens,
        "extend_lens_list": extend_lens,
    }


def fill_inputs(case, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    case["q"].copy_(
        torch.randn(case["q"].shape, generator=generator).to(
            dtype=case["q"].dtype, device=case["q"].device
        )
    )
    case["k_buffer"].copy_(
        torch.randn(case["k_buffer"].shape, generator=generator).to(
            dtype=case["k_buffer"].dtype, device=case["k_buffer"].device
        )
    )
    case["v_buffer"].copy_(
        torch.randn(case["v_buffer"].shape, generator=generator).to(
            dtype=case["v_buffer"].dtype, device=case["v_buffer"].device
        )
    )


def build_extend_tensors(case):
    k_extend = torch.empty_like(case["k_buffer"][: case["q"].shape[0]])
    v_extend = torch.empty_like(case["v_buffer"][: case["q"].shape[0]])
    output_offset = 0
    cache_offset = 0
    for prefix_len, extend_len in zip(
        case["prefix_lens_list"], case["extend_lens_list"]
    ):
        cache_start = cache_offset + prefix_len
        cache_end = cache_start + extend_len
        output_end = output_offset + extend_len
        k_extend[output_offset:output_end] = case["k_buffer"][cache_start:cache_end]
        v_extend[output_offset:output_end] = case["v_buffer"][cache_start:cache_end]
        output_offset = output_end
        cache_offset = cache_end
    return k_extend, v_extend


def reference_extend_attention(case):
    k_extend, v_extend = build_extend_tensors(case)
    output = torch.empty_like(case["q"], dtype=torch.bfloat16)
    group_size = case["q"].shape[1] // case["k_buffer"].shape[1]

    for batch_index, (prefix_len, extend_len) in enumerate(
        zip(case["prefix_lens_list"], case["extend_lens_list"])
    ):
        q_start = int(case["qo_indptr"][batch_index])
        q_end = int(case["qo_indptr"][batch_index + 1])
        prefix_start = int(case["prefix_kv_indptr"][batch_index])
        prefix_end = int(case["prefix_kv_indptr"][batch_index + 1])
        prefix_indices = case["prefix_kv_indices"][prefix_start:prefix_end]
        k_prefix = case["k_buffer"][prefix_indices]
        v_prefix = case["v_buffer"][prefix_indices]
        k_full = torch.cat([k_prefix, k_extend[q_start:q_end]], dim=0)
        v_full = torch.cat([v_prefix, v_extend[q_start:q_end]], dim=0)
        if group_size != 1:
            k_full = k_full.repeat_interleave(group_size, dim=1)
            v_full = v_full.repeat_interleave(group_size, dim=1)
        query = case["q"][q_start:q_end].to(torch.float32)
        scores = torch.einsum("qhd,khd->qhk", query, k_full.to(torch.float32)) / (
            query.shape[-1] ** 0.5
        )
        total_len = prefix_len + extend_len
        key_positions = torch.arange(total_len, device=query.device)
        query_positions = torch.arange(prefix_len, total_len, device=query.device)
        causal_mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
        scores = scores.masked_fill(~causal_mask.unsqueeze(1), float("-inf"))
        probabilities = torch.softmax(scores, dim=-1)
        output[q_start:q_end] = torch.einsum(
            "qhk,khd->qhd", probabilities, v_full.to(torch.float32)
        ).to(torch.bfloat16)
    return output


def run_reference(case):
    return reference_extend_attention(case)


def run_unified(case, output):
    extend_attention_fwd_unified(
        case["q"],
        output,
        case["k_buffer"],
        case["v_buffer"],
        1.0,
        1.0,
        case["qo_indptr"],
        case["kv_indptr"],
        case["unified_kv_indices"],
        case["prefix_lens"],
        max_len_extend=case["max_len_extend"],
        custom_mask=None,
        mask_indptr=None,
        sm_scale=None,
        logit_cap=0.0,
        is_causal=True,
    )


def time_mode(case, mode, iterations=5):
    warmup_start = time.perf_counter()
    if mode == "fresh":
        warmup_output = torch.empty_like(case["q"], dtype=torch.bfloat16)
    else:
        warmup_output = case["o_reused"]
    warmup_output.fill_(float("nan"))
    run_unified(case, warmup_output)
    torch.cuda.synchronize()
    warmup_ms = (time.perf_counter() - warmup_start) * 1000.0

    storage_samples = []
    kernel_samples = []
    for _ in range(iterations):
        storage_start = time.perf_counter()
        if mode == "fresh":
            output = torch.empty_like(case["q"], dtype=torch.bfloat16)
        else:
            output = case["o_reused"]
        output.fill_(float("nan"))
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        run_unified(case, output)
        end.record()
        torch.cuda.synchronize()
        storage_samples.append((time.perf_counter() - storage_start) * 1000.0)
        kernel_samples.append(start.elapsed_time(end))
        if torch.isnan(output).any() or not torch.isfinite(output).all():
            raise RuntimeError("sentinel remained or output was non-finite")
    return warmup_ms, storage_samples, kernel_samples


def main():
    device = torch.device("cuda")
    results = {
        "gpu": torch.cuda.get_device_name(device),
        "capability": list(torch.cuda.get_device_capability(device)),
        "timing_method": "one warmup, then host perf_counter for storage strategy (fresh allocation or reuse, sentinel fill, kernel, synchronize) and CUDA events for kernel-only; 5 samples each",
        "configs": [],
    }
    for config in CONFIGS:
        case = make_case(config, device)
        fixed_addresses = {
            key: tensor.data_ptr()
            for key, tensor in case.items()
            if isinstance(tensor, torch.Tensor)
        }
        config_result = {"name": config["name"], "batches": []}
        for seed in (1234, 5678):
            fill_inputs(case, seed)
            reference = run_reference(case)
            fresh_warmup, fresh_storage, fresh_kernel = time_mode(case, "fresh")
            reuse_warmup, reuse_storage, reuse_kernel = time_mode(case, "reuse")
            run_unified(case, case["o_reused"])
            max_diff = (case["o_reused"] - reference).abs().max().item()
            allclose = torch.allclose(case["o_reused"], reference, rtol=0.05, atol=0.05)
            config_result["batches"].append(
                {
                    "seed": seed,
                    "fresh_warmup_ms": fresh_warmup,
                    "reuse_warmup_ms": reuse_warmup,
                    "fresh_storage_ms": fresh_storage,
                    "fresh_kernel_ms": fresh_kernel,
                    "reuse_storage_ms": reuse_storage,
                    "reuse_kernel_ms": reuse_kernel,
                    "fresh_storage_median_ms": statistics.median(fresh_storage),
                    "fresh_kernel_median_ms": statistics.median(fresh_kernel),
                    "reuse_storage_median_ms": statistics.median(reuse_storage),
                    "reuse_kernel_median_ms": statistics.median(reuse_kernel),
                    "reference_allclose": allclose,
                    "reference_max_abs_diff": max_diff,
                }
            )
        current_addresses = {
            key: tensor.data_ptr()
            for key, tensor in case.items()
            if isinstance(tensor, torch.Tensor)
        }
        if current_addresses != fixed_addresses:
            raise RuntimeError("tensor address changed between batches")
        config_result["fixed_addresses"] = fixed_addresses
        results["configs"].append(config_result)

    output_path = Path(__file__).with_name("results.json")
    output_path.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    sys.exit(main())
