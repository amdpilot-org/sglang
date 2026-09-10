import json
import math
import os
import platform
import sys
import types

import torch

from sglang.srt.layers.attention.deepseek_v4_backend_hip_radix import (
    DeepseekV4HipRadixBackend,
)
from sglang.srt.layers.attention.dsv4.compress_hip import (
    capture_c4_state_windows_unified,
)
from sglang.srt.mem_cache.deepseek_v4_compress_state import CompressStatePool
from sglang.srt.mem_cache.memory_pool_host import DeepSeekV4PagedHostPool
from sglang.srt.mem_cache.unified_cache.components.swa_component import (
    _restore_state_windows,
)


def _host_pool(device_buffers, item_bytes, slot_page_size, pages, name):
    pool = DeepSeekV4PagedHostPool(
        pool_name=name,
        device_buffers=device_buffers,
        item_bytes=item_bytes,
        num_host_pages=pages,
        slot_page_size=slot_page_size,
        layout="layer_first",
        device="cpu",
        pin_memory=True,
        allocator_type="default",
    )
    pool._capture_staging = {}
    return pool


def _forward_batch(total, rid):
    return types.SimpleNamespace(
        batch_size=1,
        forward_mode=types.SimpleNamespace(is_extend=lambda: True),
        extend_prefix_lens_cpu=[0],
        extend_seq_lens_cpu=[total],
        seq_lens_cpu=torch.tensor([total], dtype=torch.int64),
        req_pool_indices=torch.tensor([rid], dtype=torch.int64),
        orig_seq_lens=torch.tensor([total], dtype=torch.int64),
    )


def _attention_continuation(swa_window, state_window, head_dim, query):
    keys = torch.cat([swa_window, state_window[:, :head_dim]], dim=0).float()
    values = keys
    scores = keys @ query / math.sqrt(head_dim)
    weights = torch.softmax(scores, dim=0)
    return weights @ values


def _ring_block(flat, boundary, ring):
    positions = torch.arange(boundary - ring, boundary, device=flat.device)
    rows = positions % ring
    block = torch.empty(ring, flat.shape[-1], device=flat.device, dtype=flat.dtype)
    block[rows] = flat[positions]
    return block


def _run_case(name, *, page, ring, total, target_slot, unrelated_slots):
    torch.manual_seed(20260910)
    device = torch.device("cuda:0")
    layers = 2
    head_dim = 16
    state_head_dim = 8
    state_ring = 8
    ratio = 4
    state_last_dim = 2 * (1 + True) * state_head_dim
    slots = 4
    dtype = torch.bfloat16

    flat_swa = [
        torch.randn(total, head_dim, device=device, dtype=dtype) for _ in range(layers)
    ]
    flat_state = [
        torch.randn(total, state_last_dim, device=device, dtype=dtype)
        for _ in range(layers)
    ]

    swa_device = [
        torch.randn(slots, ring, head_dim, device=device, dtype=dtype)
        for _ in range(layers)
    ]
    swa_bytes = [t.view(torch.uint8).reshape(slots, -1) for t in swa_device]
    swa_item_bytes = ring * head_dim * dtype.itemsize
    swa_host = _host_pool(
        swa_bytes,
        item_bytes=swa_item_bytes,
        slot_page_size=ring,
        pages=8,
        name=f"{name}-swa",
    )

    state_device = [
        torch.randn(slots, state_ring, state_last_dim, device=device, dtype=dtype)
        for _ in range(layers)
    ]
    state_bytes = [t.view(torch.uint8).reshape(slots, -1) for t in state_device]
    state_item_bytes = state_ring * state_last_dim * dtype.itemsize
    state_host = _host_pool(
        state_bytes,
        item_bytes=state_item_bytes,
        slot_page_size=state_ring,
        pages=8,
        name=f"{name}-state",
    )

    state_pools = []
    for _ in range(layers):
        state_pool = CompressStatePool(
            size=slots * state_ring,
            ring_size=state_ring,
            overlap=True,
            head_dim=state_head_dim,
            dtype=dtype,
            device=device,
            enable_memory_saver=False,
            ratio=ratio,
            swa_page_size=ring,
            state_cache_page_size=1,
        )
        state_pool.kv_score_buffer.kv_score.copy_(
            torch.arange(
                state_pool.kv_score_buffer.kv_score.numel(),
                device=device,
                dtype=torch.float32,
            ).reshape(-1, state_last_dim).to(dtype)
        )
        state_pools.append(state_pool)

    token_pool = types.SimpleNamespace(
        _swa_host_pool=swa_host,
        unified_swa_ring_size=ring,
        unified_swa_window=ring,
        start_layer=0,
        _swa_offload_page_stride=1,
        _swa_capture_bigram_key=False,
        _c4_state_host_pool=state_host,
        _c4_state_layer_index={layer: layer for layer in range(layers)},
    )
    backend = types.SimpleNamespace(token_to_kv_pool=token_pool, page_size=page)
    forward_batch = _forward_batch(total, rid=0)

    for layer in range(layers):
        DeepseekV4HipRadixBackend.capture_swa_windows(
            backend, layer, flat_swa[layer], forward_batch
        )
        capture_c4_state_windows_unified(
            backend=backend,
            state_pool=state_pools[layer],
            kv_score_input=flat_state[layer],
            forward_batch=forward_batch,
            is_indexer=False,
            layer_id=layer,
            ratio=ratio,
        )

    boundary = (total // page) * page
    swa_host_indices = swa_host._capture_staging[(0, boundary)]
    state_host_indices = state_host._capture_staging[(0, boundary)]

    unrelated_swa_before = {
        (slot, layer): swa_device[layer][slot].clone()
        for slot in unrelated_slots
        for layer in range(layers)
    }
    unrelated_state_before = [
        state_pool.kv_score_buffer.kv_score.clone() for state_pool in state_pools
    ]
    positions_for_zero = torch.arange(boundary - ratio, boundary, device=device)
    state_locs_for_zero = state_pools[0].translate_from_req_position_to_state_loc(
        torch.tensor([target_slot], device=device), positions_for_zero
    )

    for layer in range(layers):
        swa_device[layer][target_slot].fill_(0)
    for state_pool in state_pools:
        state_pool.kv_score_buffer.kv_score[state_locs_for_zero].fill_(0)

    device_indices = torch.arange(
        target_slot * ring, (target_slot + 1) * ring, dtype=torch.int64
    )
    swa_host.load_to_device_all_layer(
        None, swa_host_indices, device_indices, io_backend="direct"
    )

    node = types.SimpleNamespace(
        _c4_state_host_value=state_host_indices,
        _c4_indexer_state_host_value=None,
        _swa_state_B=boundary,
    )
    restorer = types.SimpleNamespace(
        _c4_state_layer_index={layer: layer for layer in range(layers)},
        _c4_state_host_pool=state_host,
        _c4_indexer_state_host_pool=None,
        _compress_state_pools=state_pools,
        _indexer_compress_state_pools=None,
    )
    _restore_state_windows(restorer, node, target_slot)
    torch.cuda.synchronize()

    expected_swa = [
        _ring_block(flat_swa[layer], boundary, ring) for layer in range(layers)
    ]
    restored_swa = [swa_device[layer][target_slot].clone() for layer in range(layers)]

    positions = torch.arange(boundary - ratio, boundary, device=device)
    state_locs = state_pools[0].translate_from_req_position_to_state_loc(
        torch.tensor([target_slot], device=device), positions
    )
    expected_state = [
        flat_state[layer][boundary - ratio : boundary].clone()
        for layer in range(layers)
    ]
    restored_state = [
        state_pools[layer].kv_score_buffer.kv_score[state_locs].clone()
        for layer in range(layers)
    ]

    swa_equal = all(
        torch.equal(a.view(torch.uint8), b.view(torch.uint8))
        for a, b in zip(restored_swa, expected_swa)
    )
    state_equal = all(
        torch.equal(a.view(torch.uint8), b.view(torch.uint8))
        for a, b in zip(restored_state, expected_state)
    )

    unrelated_swa_equal = all(
        torch.equal(
            swa_device[layer][slot].view(torch.uint8),
            unrelated_swa_before[(slot, layer)].view(torch.uint8),
        )
        for slot in unrelated_slots
        for layer in range(layers)
    )
    target_rows = set(state_locs.tolist())
    unrelated_rows = [
        row
        for row in range(state_pools[0].kv_score_buffer.kv_score.shape[0])
        if row not in target_rows
    ]
    unrelated_state_equal = all(
        torch.equal(
            state_pool.kv_score_buffer.kv_score[unrelated_rows].view(torch.uint8),
            before[unrelated_rows].view(torch.uint8),
        )
        for state_pool, before in zip(state_pools, unrelated_state_before)
    )

    query = torch.arange(head_dim, device=device, dtype=torch.float32) / head_dim
    cold_attention = _attention_continuation(
        expected_swa[0], expected_state[0], head_dim, query
    )
    restored_attention = _attention_continuation(
        restored_swa[0], restored_state[0], head_dim, query
    )
    attention_equal = torch.equal(restored_attention, cold_attention)

    result = {
        "case": name,
        "page": page,
        "ring": ring,
        "boundary": boundary,
        "total": total,
        "tail": total - boundary,
        "swa_bytes_equal": swa_equal,
        "state_bytes_equal": state_equal,
        "unrelated_swa_unchanged": unrelated_swa_equal,
        "unrelated_state_unchanged": unrelated_state_equal,
        "attention_equal": attention_equal,
        "attention_max_abs_diff": float(
            (restored_attention - cold_attention).abs().max().item()
        ),
    }
    return result


def main():
    assert torch.cuda.is_available()
    assert torch.cuda.get_device_name(0) == "AMD Instinct MI300X"
    assert torch.cuda.get_device_capability(0) == (9, 4)

    cases = [
        ("aligned-prefix-partial-tail", 64, 32, 96, 2, [0, 1, 3]),
        ("short-prefix-partial-tail", 32, 16, 50, 2, [0, 1, 3]),
        ("nondividing-ring-partial-tail", 64, 20, 80, 2, [0, 1, 3]),
    ]
    results = []
    for name, page, ring, total, target, unrelated in cases:
        result = _run_case(
            name,
            page=page,
            ring=ring,
            total=total,
            target_slot=target,
            unrelated_slots=unrelated,
        )
        results.append(result)
        print(json.dumps(result, sort_keys=True))

    summary = {
        "gpu": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "torch": torch.__version__,
        "python": sys.version,
        "platform": platform.platform(),
        "cases": results,
    }
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
