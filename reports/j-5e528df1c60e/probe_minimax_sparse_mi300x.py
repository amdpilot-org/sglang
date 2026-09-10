from __future__ import annotations

import argparse
import json
import os
import time
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.minimax_sparse_backend import (
    MiniMaxSparseAttnBackend,
)
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.memory_pool import MiniMaxSparseKVPool, ReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import ForwardMode


DEVICE = "cuda"
DTYPE = torch.bfloat16
BLOCK_SIZE_K = 16
TOPK_BLOCKS = 4
INIT_BLOCKS = 1
LOCAL_BLOCKS = 2


def make_backend(speculative_algorithm: str | None) -> MiniMaxSparseAttnBackend:
    sparse_config = {
        "sparse_index_dim": 16,
        "sparse_attention_freq": [0, 1],
        "sparse_disable_index_value": [0, 1],
        "sparse_score_type": "max",
        "sparse_block_size": BLOCK_SIZE_K,
        "sparse_init_block": INIT_BLOCKS,
        "sparse_local_block": LOCAL_BLOCKS,
        "sparse_topk_blocks": TOPK_BLOCKS,
        "sparse_num_index_heads": 1,
    }
    hf_config = SimpleNamespace(sparse_attention_config=sparse_config)
    model_config = SimpleNamespace(
        context_len=64,
        hf_config=hf_config,
        num_attention_heads=2,
    )
    server_args = SimpleNamespace(
        kv_cache_dtype="auto",
        attention_backend="trtllm_mha",
    )
    kv_pool = MiniMaxSparseKVPool(
        size=128,
        page_size=1,
        dtype=DTYPE,
        head_num=1,
        head_dim=32,
        idx_head_dim=16,
        dense_layer_ids=[0],
        sparse_layer_ids=[1],
        disable_value_sparse_layer_ids=[1],
        device=DEVICE,
        start_layer=0,
        end_layer=2,
    )
    req_pool = ReqToTokenPool(
        size=2,
        max_context_len=64,
        device=DEVICE,
        enable_memory_saver=False,
    )
    runner = SimpleNamespace(
        token_to_kv_pool=kv_pool,
        req_to_token_pool=req_pool,
        model_config=model_config,
        server_args=server_args,
    )
    backend_module = MiniMaxSparseAttnBackend.__module__
    import importlib

    module = importlib.import_module(backend_module)
    original_get_spec = module.get_spec
    module.get_spec = lambda: SimpleNamespace(
        speculative_algorithm=speculative_algorithm,
        speculative_num_draft_tokens=4 if speculative_algorithm else None,
    )
    try:
        return MiniMaxSparseAttnBackend(runner)
    finally:
        module.get_spec = original_get_spec


def make_layer() -> RadixAttention:
    layer = RadixAttention(
        num_heads=2,
        head_dim=32,
        scaling=32**-0.5,
        num_kv_heads=1,
        layer_id=1,
    )
    layer.q_scale_float = None
    layer.k_scale_float = None
    layer.v_scale_float = None
    layer.idx_q_scale_float = None
    layer.idx_k_scale_float = None
    layer.idx_v_scale_float = None
    return layer


def make_batch(
    mode: ForwardMode,
    extend_lens: list[int] | None,
) -> tuple[SimpleNamespace, torch.Tensor, torch.Tensor]:
    torch.manual_seed(1234)
    req_to_token = torch.zeros((2, 64), dtype=torch.int32, device=DEVICE)
    slot = 1
    slots: list[int] = []
    lengths = [17, 19] if mode == ForwardMode.DECODE else extend_lens
    for request in range(2):
        length = lengths[request]
        positions = torch.arange(length, device=DEVICE, dtype=torch.int32)
        request_slots = torch.arange(
            slot, slot + length, device=DEVICE, dtype=torch.int32
        )
        req_to_token[request, :length] = request_slots
        if mode == ForwardMode.DECODE:
            slots.append(int(request_slots[-1].item()))
        else:
            slots.extend(request_slots.tolist())
        slot += length

    out_cache_loc = torch.tensor(slots, dtype=torch.int64, device=DEVICE)
    seq_lens = torch.tensor(lengths, dtype=torch.int32, device=DEVICE)
    if mode in {ForwardMode.DECODE, ForwardMode.TARGET_VERIFY}:
        extend_seq_lens = None
        extend_seq_lens_cpu = None
        extend_prefix_lens = None
    else:
        extend_seq_lens = torch.tensor(extend_lens, dtype=torch.int32, device=DEVICE)
        extend_seq_lens_cpu = extend_lens
        extend_prefix_lens = torch.zeros(2, dtype=torch.int32, device=DEVICE)

    batch = SimpleNamespace(
        forward_mode=mode,
        req_pool_indices=torch.tensor([1, 2], dtype=torch.int64, device=DEVICE),
        seq_lens=seq_lens,
        seq_lens_cpu=seq_lens.cpu(),
        out_cache_loc=out_cache_loc,
        extend_seq_lens=extend_seq_lens,
        extend_seq_lens_cpu=extend_seq_lens_cpu,
        extend_prefix_lens=extend_prefix_lens,
        minimax_m3_precached_sparse_layers=None,
    )
    return batch, req_to_token, out_cache_loc


def torch_reference(
    backend: MiniMaxSparseAttnBackend,
    layer: RadixAttention,
    mode: ForwardMode,
    extend_lens: list[int] | None,
    q: torch.Tensor,
    idx_q: torch.Tensor,
) -> torch.Tensor:
    batch, _, _ = make_batch(mode, extend_lens)
    k_cache, _ = backend.kv_pool.get_kv_buffer(layer.layer_id)
    idx_k_cache = backend.kv_pool.get_index_k_buffer(layer.layer_id)
    outputs = []
    idx_scale = backend.idx_head_dim**-0.5

    for request, req_index in enumerate(batch.req_pool_indices.tolist()):
        seq_len = int(batch.seq_lens[request].item())
        query_count = 1 if mode == ForwardMode.DECODE else seq_len
        for query_offset in range(query_count):
            token_index = request if mode == ForwardMode.DECODE else (
                sum(extend_lens[:request]) + query_offset
            )
            attended_len = seq_len if mode == ForwardMode.DECODE else query_offset + 1
            positions = torch.arange(attended_len, device=DEVICE)
            slots = backend.req_to_token[req_index, positions].long()
            idx_k = idx_k_cache[slots, 0].float()
            scores = (
                idx_q[token_index, 0].float() @ idx_k.transpose(0, 1)
            ) * idx_scale
            block_count = (attended_len + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
            block_scores = torch.empty(block_count, device=DEVICE)
            for block in range(block_count):
                start = block * BLOCK_SIZE_K
                end = min(start + BLOCK_SIZE_K, attended_len)
                block_scores[block] = scores[start:end].max()
            block_scores[:INIT_BLOCKS] = 1e30
            local_start = max(0, block_count - LOCAL_BLOCKS)
            block_scores[local_start:block_count] = 1e29
            selected_blocks = torch.topk(
                block_scores, min(TOPK_BLOCKS, block_count)
            ).indices.tolist()
            selected_positions = torch.cat(
                [
                    torch.arange(
                        block * BLOCK_SIZE_K,
                        min((block + 1) * BLOCK_SIZE_K, attended_len),
                        device=DEVICE,
                )
                    for block in selected_blocks
                ]
            )
            selected_slots = backend.req_to_token[
                req_index, selected_positions
            ].long()
            selected_k = k_cache[selected_slots, 0].float()
            selected_v = backend.kv_pool.get_value_buffer(layer.layer_id)[
                selected_slots, 0
            ].float()
            query = q[token_index]
            logits = (query.float() @ selected_k.transpose(0, 1)) * layer.scaling
            weights = torch.softmax(logits, dim=-1)
            outputs.append(weights @ selected_v)

    return torch.stack(outputs).reshape(-1, layer.tp_q_head_num * layer.v_head_dim)


def run_supported(mode: ForwardMode, extend_lens: list[int] | None) -> dict:
    backend = make_backend(None)
    backend.req_to_token.zero_()
    layer = make_layer()
    batch, req_to_token, _ = make_batch(mode, extend_lens)
    backend.req_to_token[: req_to_token.shape[0]].copy_(req_to_token)
    token_count = batch.out_cache_loc.numel()
    if mode == ForwardMode.DECODE:
        for request in range(2):
            seq_len = int(batch.seq_lens[request].item())
            previous_slots = req_to_token[request, : seq_len - 1].long()
            if previous_slots.numel():
                previous_count = previous_slots.numel()
                backend.kv_pool.set_fused_kv_index_buffer(
                    layer,
                    previous_slots,
                    torch.randn(previous_count, 1, 32, dtype=DTYPE, device=DEVICE),
                    torch.randn(previous_count, 1, 32, dtype=DTYPE, device=DEVICE),
                    torch.randn(previous_count, 1, 16, dtype=DTYPE, device=DEVICE),
                    None,
                )
    q = torch.randn(token_count, 2, 32, dtype=DTYPE, device=DEVICE)
    k = torch.randn(token_count, 1, 32, dtype=DTYPE, device=DEVICE)
    v = torch.randn(token_count, 1, 32, dtype=DTYPE, device=DEVICE)
    idx_q = torch.randn(token_count, 1, 16, dtype=DTYPE, device=DEVICE)
    idx_k = torch.randn(token_count, 1, 16, dtype=DTYPE, device=DEVICE)
    idx_v = None

    backend.init_forward_metadata_out_graph(batch)
    started = time.perf_counter()
    if mode == ForwardMode.DECODE:
        idx_out, output = backend.forward_decode(
            q, k, v, layer, batch, idx_q=idx_q, idx_k=idx_k, idx_v=idx_v
        )
    else:
        idx_out, output = backend.forward_extend(
            q, k, v, layer, batch, idx_q=idx_q, idx_k=idx_k, idx_v=idx_v
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    reference = torch_reference(backend, layer, mode, extend_lens, q, idx_q)
    difference = (output.float() - reference.float()).abs()
    return {
        "mode": mode.name,
        "extend_seq_lens": None if mode == ForwardMode.DECODE else extend_lens,
        "output_shape": list(output.shape),
        "idx_out_is_none": idx_out is None,
        "max_abs_diff": float(difference.max().item()),
        "mean_abs_diff": float(difference.mean().item()),
        "seconds": elapsed,
    }


def run_target_verify_negative() -> dict:
    started = time.perf_counter()
    try:
        backend = make_backend("EAGLE3")
    except NotImplementedError as error:
        return {
            "mode": "TARGET_VERIFY",
            "result": "early_diagnostic",
            "message": str(error),
            "seconds": time.perf_counter() - started,
        }

    backend.req_to_token.zero_()
    layer = make_layer()
    batch, req_to_token, _ = make_batch(ForwardMode.TARGET_VERIFY, [4, 4])
    backend.req_to_token[: req_to_token.shape[0]].copy_(req_to_token)
    token_count = 8
    q = torch.randn(token_count, 2, 32, dtype=DTYPE, device=DEVICE)
    k = torch.randn(token_count, 1, 32, dtype=DTYPE, device=DEVICE)
    v = torch.randn(token_count, 1, 32, dtype=DTYPE, device=DEVICE)
    idx_q = torch.randn(token_count, 1, 16, dtype=DTYPE, device=DEVICE)
    idx_k = torch.randn(token_count, 1, 16, dtype=DTYPE, device=DEVICE)
    try:
        backend.init_forward_metadata_out_graph(batch)
        backend.forward_extend(
            q,
            k,
            v,
            layer,
            batch,
            idx_q=idx_q,
            idx_k=idx_k,
            idx_v=None,
        )
        torch.cuda.synchronize()
        return {
            "mode": "TARGET_VERIFY",
            "result": "unexpected_success",
            "seconds": time.perf_counter() - started,
        }
    except Exception as error:
        return {
            "mode": "TARGET_VERIFY",
            "result": "late_failure",
            "error_type": type(error).__name__,
            "message": str(error),
            "seconds": time.perf_counter() - started,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    results = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "decode": run_supported(ForwardMode.DECODE, None),
        "extend": run_supported(ForwardMode.EXTEND, [3, 2]),
        "target_verify": run_target_verify_negative(),
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
