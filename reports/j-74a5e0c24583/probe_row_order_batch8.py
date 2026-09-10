import torch
from sglang.kernels.ops.speculative.dspark import dspark_verify_window
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout

device = torch.device("cuda")
base_lens = [1, 4, 2, 5, 3, 1, 4, 2]
patterns = [
    ("base", list(range(8))),
    ("perm", [2, 0, 3, 1, 5, 7, 4, 6]),
    ("inverse", [1, 3, 0, 2, 6, 4, 7, 5]),
    ("reverse", list(range(7, -1, -1))),
]
bs = len(base_lens)
gamma = 5
stride = 5
graph_num_tokens = 32
dim = 3
fill_value = -1.0

def row_reference(order):
    reqs = []
    offs = []
    for new_request_id, old_request_id in enumerate(order):
        length = base_lens[old_request_id]
        reqs.extend([new_request_id] * length)
        offs.extend(range(length))
    valid = [True] * len(reqs)
    reqs += [bs] * (graph_num_tokens - len(reqs))
    offs += [0] * (graph_num_tokens - len(offs))
    valid += [False] * (graph_num_tokens - len(valid))
    return (
        torch.tensor(reqs, dtype=torch.int64, device=device),
        torch.tensor(offs, dtype=torch.int64, device=device),
        torch.tensor(valid, dtype=torch.bool, device=device),
    )

def ids_reference(order, block_ids, tokens):
    out = []
    for new_request_id, old_request_id in enumerate(order):
        length = base_lens[old_request_id]
        out.append(int(block_ids[new_request_id, 0]))
        out.extend(int(tokens[new_request_id, j]) for j in range(length - 1))
    out.extend([0] * (graph_num_tokens - len(out)))
    return torch.tensor(out, dtype=torch.int64, device=device)

def compact_for(order):
    data = []
    for old_request_id in order:
        length = base_lens[old_request_id]
        for within in range(length):
            data.append(torch.full((dim,), float(old_request_id * 100 + within), device=device))
    while len(data) < graph_num_tokens:
        data.append(torch.zeros(dim, device=device))
    return torch.stack(data)

def scatter_reference(order, compact):
    out = torch.full((bs * stride, dim), fill_value, device=device)
    cursor = 0
    for new_request_id, old_request_id in enumerate(order):
        length = base_lens[old_request_id]
        out[new_request_id * stride : new_request_id * stride + length] = compact[cursor : cursor + length]
        cursor += length
    return out

verify_lens = torch.tensor(base_lens, dtype=torch.int32, device=device)
layout = RaggedVerifyLayout.from_verify_lens_device(
    verify_lens=verify_lens, graph_num_tokens=graph_num_tokens
)
block_ids = torch.arange(bs * gamma, device=device, dtype=torch.int64).reshape(bs, gamma)
tokens = block_ids + 1000
compact = compact_for(patterns[0][1])

# Warm up all three operations outside capture.
dspark_verify_window.CompactRowIndex.triton(
    verify_lens=verify_lens, padded_total=graph_num_tokens, device=device
)
dspark_verify_window.CompactVerifyIds.triton(
    draft_block_ids=block_ids,
    draft_tokens=tokens,
    layout=layout,
    device=device,
)
dspark_verify_window.ScatterCompactToStrided.triton(
    compact=compact,
    layout=layout,
    fill_value=fill_value,
    verify_num_draft_tokens=stride,
)
torch.cuda.synchronize()

stream = torch.cuda.Stream()
stream.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(stream):
    for _ in range(3):
        dspark_verify_window.CompactRowIndex.triton(
            verify_lens=verify_lens, padded_total=graph_num_tokens, device=device
        )
        dspark_verify_window.CompactVerifyIds.triton(
            draft_block_ids=block_ids,
            draft_tokens=tokens,
            layout=layout,
            device=device,
        )
        dspark_verify_window.ScatterCompactToStrided.triton(
            compact=compact,
            layout=layout,
            fill_value=fill_value,
            verify_num_draft_tokens=stride,
        )
torch.cuda.current_stream().wait_stream(stream)

row_graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(row_graph):
    row_req, row_within, row_valid = dspark_verify_window.CompactRowIndex.triton(
        verify_lens=verify_lens, padded_total=graph_num_tokens, device=device
    )

ids_graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(ids_graph):
    ids_replay = dspark_verify_window.CompactVerifyIds.triton(
        draft_block_ids=block_ids,
        draft_tokens=tokens,
        layout=layout,
        device=device,
    )

scatter_graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(scatter_graph):
    scatter_replay = dspark_verify_window.ScatterCompactToStrided.triton(
        compact=compact,
        layout=layout,
        fill_value=fill_value,
        verify_num_draft_tokens=stride,
    )
torch.cuda.synchronize()

for cycle, (name, order) in enumerate(patterns, 1):
    verify_lens.copy_(torch.tensor([base_lens[i] for i in order], dtype=torch.int32, device=device))
    compact.copy_(compact_for(order))

    row_graph.replay()
    torch.cuda.synchronize()
    exp_req, exp_within, exp_valid = row_reference(order)
    row_ok = (
        torch.equal(row_req, exp_req)
        and torch.equal(row_within, exp_within)
        and torch.equal(row_valid, exp_valid)
    )

    ids_graph.replay()
    torch.cuda.synchronize()
    exp_ids = ids_reference(order, block_ids, tokens)
    ids_ok = torch.equal(ids_replay, exp_ids)

    scatter_graph.replay()
    torch.cuda.synchronize()
    exp_scatter = scatter_reference(order, compact)
    scatter_ok = torch.equal(scatter_replay, exp_scatter)

    print(f"cycle={cycle} pattern={name} row_ok={row_ok} ids_ok={ids_ok} scatter_ok={scatter_ok}")
    if not (row_ok and ids_ok and scatter_ok):
        if not row_ok:
            print("row_req", row_req.tolist())
            print("exp_req", exp_req.tolist())
            print("row_within", row_within.tolist())
            print("exp_within", exp_within.tolist())
            print("row_valid", row_valid.tolist())
            print("exp_valid", exp_valid.tolist())
        if not ids_ok:
            print("ids", ids_replay.tolist())
            print("exp_ids", exp_ids.tolist())
        if not scatter_ok:
            print("scatter", scatter_replay.tolist())
            print("exp_scatter", exp_scatter.tolist())
        raise SystemExit(1)
print("BATCH8_ROW_ORDER_PROBE_OK")
