import types

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout

device = torch.device("cuda")
batch_size, graph_tokens = 48, 640
verify_lens = [12] * 47 + [11]
qo_indptr_cpu = torch.tensor(
    [0] + list(torch.tensor(verify_lens).cumsum(0).tolist()), dtype=torch.int32
)
real_tokens = sum(verify_lens)
positions_cpu = torch.full((graph_tokens,), -17, dtype=torch.int64)
reference = torch.zeros(graph_tokens, dtype=torch.int64)
deltas = []
for request, (start, end) in enumerate(zip(qo_indptr_cpu[:-1], qo_indptr_cpu[1:])):
    start, end = int(start), int(end)
    positions_cpu[start:end] = torch.arange(
        1000 + request * 100, 1000 + request * 100 + end - start
    )
    delta = request * 7 - 50
    deltas.append(delta)
    reference[start:end] = positions_cpu[start:end] + delta

layout = RaggedVerifyLayout(
    verify_lens=torch.tensor(verify_lens, dtype=torch.int32, device=device),
    graph_num_tokens=graph_tokens,
    extend_start_loc=qo_indptr_cpu[:-1].to(device),
    qo_indptr_device=qo_indptr_cpu.to(device),
)
forward_batch = ForwardBatch.__new__(ForwardBatch)
forward_batch.seq_lens = torch.ones(batch_size, dtype=torch.int32, device=device)
batch = types.SimpleNamespace(
    multimodal_inputs=[
        types.SimpleNamespace(mrope_position_delta=torch.tensor([[delta]]))
        for delta in deltas
    ],
    spec_info=types.SimpleNamespace(
        positions=positions_cpu.to(device), ragged_verify_layout=layout
    ),
)
forward_batch.compute_spec_mrope_positions(types.SimpleNamespace(device=device), batch)
torch.cuda.synchronize()
actual = forward_batch.mrope_positions.cpu()
torch.testing.assert_close(actual, reference.repeat(3, 1))
print(
    {
        "device": torch.cuda.get_device_name(0),
        "batch_size": batch_size,
        "graph_tokens": graph_tokens,
        "real_tokens": real_tokens,
        "padding_tokens": graph_tokens - real_tokens,
        "shape": tuple(actual.shape),
        "max_abs_error": int((actual - reference).abs().max()),
    }
)
