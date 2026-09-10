import json
import math
import time

import torch

from sglang.kernels.ops.attention.decode_attention import (
    decode_attention_fwd_grouped,
    decode_attention_fwd_normal,
)


def temperature_scale(position, threshold):
    if position <= threshold:
        return 1.0
    return math.log2(position) / math.log2(threshold)


def make_inputs(batch_count, query_heads, kv_heads, sequence_lengths, seed):
    torch.manual_seed(seed)
    device = "cuda"
    dtype = torch.bfloat16
    head_dim = 64
    value_dim = 64
    total_tokens = sum(sequence_lengths)
    queries = torch.randn(
        batch_count, query_heads, head_dim, dtype=dtype, device=device
    )
    keys = torch.randn(
        total_tokens, kv_heads, head_dim, dtype=dtype, device=device
    )
    values = torch.randn(
        total_tokens, kv_heads, value_dim, dtype=dtype, device=device
    )
    kv_indptr = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(sequence_lengths), 0)),
        dtype=torch.int32,
        device=device,
    )
    kv_indices = torch.arange(total_tokens, device=device)
    assert torch.isfinite(queries.float()).all()
    assert torch.isfinite(keys.float()).all()
    assert torch.isfinite(values.float()).all()
    return queries, keys, values, kv_indptr, kv_indices


def reference_partials(
    queries,
    keys,
    values,
    kv_indptr,
    query_heads,
    kv_heads,
    sequence_lengths,
    threshold,
):
    head_dim = queries.shape[-1]
    value_dim = values.shape[-1]
    sm_scale = 1.0 / math.sqrt(head_dim)
    split_count = 1
    batch_count = queries.shape[0]
    lse = torch.empty(
        batch_count, query_heads, split_count, dtype=torch.float32, device=queries.device
    )
    split_outputs = torch.empty(
        batch_count,
        query_heads,
        split_count,
        value_dim,
        dtype=torch.float32,
        device=queries.device,
    )
    expected_scales = []
    kv_group_num = query_heads // kv_heads
    for batch_index, sequence_length in enumerate(sequence_lengths):
        position = sequence_length - 1
        scale = temperature_scale(position, threshold)
        expected_scales.append(scale)
        token_start = int(kv_indptr[batch_index])
        token_end = int(kv_indptr[batch_index + 1])
        for head_index in range(query_heads):
            kv_head = head_index // kv_group_num
            query = queries[batch_index, head_index].float()
            key = keys[token_start:token_end, kv_head].float()
            value = values[token_start:token_end, kv_head].float()
            logits = (key @ query) * sm_scale * scale
            lse[batch_index, head_index, 0] = torch.logsumexp(logits, dim=0)
            split_outputs[batch_index, head_index, 0] = (
                torch.softmax(logits, dim=0) @ value
            )
    return lse, split_outputs, expected_scales


def run_grouped_probe():
    batch_count = 3
    query_heads = 18
    kv_heads = 2
    sequence_lengths = [4, 5, 6]
    threshold = 4
    max_splits = 1
    queries, keys, values, kv_indptr, kv_indices = make_inputs(
        batch_count, query_heads, kv_heads, sequence_lengths, 32942
    )
    split_count = torch.ones(batch_count, dtype=torch.int32, device="cuda")
    outputs = torch.empty_like(queries)
    split_outputs = torch.empty(
        batch_count,
        query_heads,
        max_splits,
        values.shape[-1],
        dtype=torch.float32,
        device="cuda",
    )
    lse_size = batch_count * query_heads * max_splits
    lse_storage = torch.full((lse_size + 16,), -777.0, dtype=torch.float32, device="cuda")
    lse = lse_storage[:lse_size].view(batch_count, query_heads, max_splits)
    start_time = time.perf_counter()
    decode_attention_fwd_grouped(
        queries,
        keys,
        values,
        outputs,
        kv_indptr,
        kv_indices,
        split_outputs,
        lse,
        split_count,
        max_splits,
        1.0 / math.sqrt(queries.shape[-1]),
        1.0,
        xai_temperature_len=threshold,
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time
    reference_lse, reference_splits, expected_scales = reference_partials(
        queries,
        keys,
        values,
        kv_indptr,
        query_heads,
        kv_heads,
        sequence_lengths,
        threshold,
    )
    return {
        "elapsed_seconds": elapsed,
        "query_positions": [length - 1 for length in sequence_lengths],
        "expected_scales": expected_scales,
        "max_abs_lse_diff": float((lse - reference_lse).abs().max()),
        "max_abs_split_output_diff": float(
            (split_outputs - reference_splits).abs().max()
        ),
        "guard_changed_count": int((lse_storage[lse_size:] != -777.0).sum()),
        "all_finite": bool(
            torch.isfinite(lse).all() and torch.isfinite(split_outputs).all()
        ),
    }


def run_normal_probe(threshold):
    batch_count = 3 if threshold > 1 else 1
    query_heads = 4
    kv_heads = 4
    sequence_lengths = [4, 5, 6] if threshold > 1 else [3]
    max_splits = 1
    queries, keys, values, kv_indptr, kv_indices = make_inputs(
        batch_count, query_heads, kv_heads, sequence_lengths, 32943
    )
    split_count = torch.ones(batch_count, dtype=torch.int32, device="cuda")
    outputs = torch.empty_like(queries)
    split_outputs = torch.empty(
        batch_count,
        query_heads,
        max_splits,
        values.shape[-1],
        dtype=torch.float32,
        device="cuda",
    )
    lse = torch.empty(
        batch_count, query_heads, max_splits, dtype=torch.float32, device="cuda"
    )
    start_time = time.perf_counter()
    decode_attention_fwd_normal(
        queries,
        keys,
        values,
        outputs,
        kv_indptr,
        kv_indices,
        split_outputs,
        lse,
        split_count,
        max_splits,
        1.0 / math.sqrt(queries.shape[-1]),
        1.0,
        xai_temperature_len=threshold,
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time
    result = {
        "elapsed_seconds": elapsed,
        "query_positions": [length - 1 for length in sequence_lengths],
        "all_finite": bool(
            torch.isfinite(lse).all() and torch.isfinite(split_outputs).all()
        ),
        "lse_nonfinite_count": int((~torch.isfinite(lse)).sum()),
        "split_nonfinite_count": int((~torch.isfinite(split_outputs)).sum()),
    }
    if threshold > 1:
        reference_lse, reference_splits, expected_scales = reference_partials(
            queries,
            keys,
            values,
            kv_indptr,
            query_heads,
            kv_heads,
            sequence_lengths,
            threshold,
        )
        result.update(
            {
                "expected_scales": expected_scales,
                "max_abs_lse_diff": float((lse - reference_lse).abs().max()),
                "max_abs_split_output_diff": float(
                    (split_outputs - reference_splits).abs().max()
                ),
                "split_output_allclose_atol_0.02_rtol_0.02": bool(
                    torch.allclose(
                        split_outputs, reference_splits, atol=0.02, rtol=0.02
                    )
                ),
            }
        )
    else:
        result["reference_supported"] = False
        result["reference_error"] = "float division by zero: log2(threshold) == 0"
    return result


def main():
    assert torch.cuda.device_count() == 1
    result = {
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
        },
        "grouped_decode": run_grouped_probe(),
        "normal_decode_threshold_4": run_normal_probe(4),
        "normal_decode_threshold_1": run_normal_probe(1),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
