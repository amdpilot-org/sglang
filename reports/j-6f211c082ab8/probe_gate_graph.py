import json
import os
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, "/job/sglang/python")
from sglang.srt.runtime_context import get_parallel

parallel_ctx = get_parallel().override(
    tp_size=1,
    tp_rank=0,
    pp_size=1,
    pp_rank=0,
    attn_tp_size=1,
    attn_tp_rank=0,
    attn_cp_size=1,
    attn_cp_rank=0,
    attn_dp_size=1,
    attn_dp_rank=0,
    moe_ep_size=1,
    moe_ep_rank=0,
    moe_dp_size=1,
    moe_dp_rank=0,
    moe_tp_size=1,
    moe_tp_rank=0,
)
parallel_ctx.__enter__()

from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

server_args = ServerArgs(model_path="dummy")
server_args.enable_dp_attention = False
set_global_server_args_for_scheduler(server_args)

import sglang.srt.models.deepseek_v2 as deepseek_v2
from sglang.srt.layers import communicator as layer_communicator

layer_communicator.get_moe_cp_size = lambda: 1
deepseek_v2.get_pp_group = lambda: SimpleNamespace(
    is_first_rank=True,
    is_last_rank=True,
    rank_in_group=0,
    world_size=1,
)
from sglang.srt.models.deepseek_v2 import DeepseekV2Model
from sglang.srt.layers.attention.dsa.dsa_indexer import DUAL_STREAM_TOKEN_THRESHOLD
from sglang.srt.model_executor.runner import get_is_capture_mode
from sglang.srt.model_executor.runner_utils.capture_mode import model_capture_mode
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    is_in_breakable_cuda_graph,
)

config = SimpleNamespace(
    architectures=["DeepseekV3ForCausalLM"],
    vocab_size=128,
    hidden_size=128,
    intermediate_size=128,
    hidden_act="silu",
    num_hidden_layers=1,
    num_attention_heads=8,
    first_k_dense_replace=0,
    pad_token_id=0,
    rms_norm_eps=1e-5,
    rope_theta=10000.0,
    rope_parameters={"rope_theta": 10000.0, "rope_type": "default"},
    rope_scaling=None,
    max_position_embeddings=128,
    q_lora_rank=64,
    kv_lora_rank=64,
    qk_nope_head_dim=64,
    qk_rope_head_dim=32,
    v_head_dim=64,
    index_topk=64,
    index_head_dim=64,
    index_n_heads=8,
    index_kpool=2,
    index_kpool_compress=True,
    index_kpool_always_select_tail=True,
    n_routed_experts=None,
)
torch.manual_seed(20260910)
torch.set_default_dtype(torch.bfloat16)
with torch.device("cuda"):
    model = DeepseekV2Model(config)
indexer = model.layers[0].self_attn.indexer
assert type(indexer).__name__ == "IndexerKPool"

for parameter in indexer.wq_b.parameters():
    parameter.data.normal_(0.0, 0.03)
for parameter in indexer.wk.parameters():
    parameter.data.normal_(0.0, 0.03)
for parameter in indexer.weights_proj.parameters():
    parameter.data.normal_(0.0, 0.03)
for parameter in indexer.k_norm.parameters():
    parameter.data.fill_(1.0)
indexer.index_kpool_compress_ape.data.normal_(0.0, 0.03)
indexer.index_kpool_compress_gate.data.normal_(0.0, 0.03)

device = torch.device("cuda")
tokens = 8
x = torch.randn(tokens, 128, dtype=torch.bfloat16, device=device)
q_lora = torch.randn(tokens, 64, dtype=torch.bfloat16, device=device)
positions = torch.arange(tokens, dtype=torch.int64, device=device)
forward_batch = SimpleNamespace(
    forward_mode=SimpleNamespace(
        is_decode_or_idle=lambda: True,
        is_extend_without_speculative=lambda: False,
    )
)

# Diagnostic-only: HIP construction intentionally omits this CUDA-only tuning
# attribute. Supplying it tests the underlying primitive without changing the gate.
half_device_sm_count_missing_before_diagnostic = not hasattr(
    indexer, "half_device_sm_count"
)
if forced_diagnostic_attr := half_device_sm_count_missing_before_diagnostic:
    indexer.half_device_sm_count = 64

with model_capture_mode():
    runtime_gate = (
        indexer.alt_stream is not None
        and get_is_capture_mode()
        and q_lora.shape[0] > 0
        and q_lora.shape[0] <= DUAL_STREAM_TOKEN_THRESHOLD
        and not (
            is_in_breakable_cuda_graph()
            and forward_batch.forward_mode.is_extend_without_speculative()
        )
    )

class RecordingStreamContext:
    def __init__(self, stream):
        self.stream = stream
    def __enter__(self):
        if self.stream is indexer.alt_stream:
            RecordingStreamContext.entries.append(self.stream)
        return self.stream.__enter__()
    def __exit__(self, *args):
        return self.stream.__exit__(*args)
RecordingStreamContext.entries = []

original_stream_context = torch.cuda.stream
torch.cuda.stream = RecordingStreamContext
try:
    single = indexer._get_q_k_bf16(
        q_lora, x, positions, runtime_gate, forward_batch
    )
    torch.cuda.synchronize()
    runtime_gate_entries = list(RecordingStreamContext.entries)
    RecordingStreamContext.entries.clear()

    forced_dual_available = indexer.alt_stream is not None
    forced_entries = []
    dual = None
    if forced_dual_available:
        dual = indexer._get_q_k_bf16(
            q_lora, x, positions, True, forward_batch
        )
        torch.cuda.synchronize()
        forced_entries = list(RecordingStreamContext.entries)
        RecordingStreamContext.entries.clear()
finally:
    torch.cuda.stream = original_stream_context

def compare(a, b):
    return {
        "equal": bool(torch.equal(a, b)),
        "max_abs_diff": float((a.float() - b.float()).abs().max().item()),
        "allclose_bf16": bool(torch.allclose(a, b, atol=2e-2, rtol=2e-2)),
    }

graph_result = None
if forced_dual_available:
    static_query = torch.empty_like(single[0])
    static_key = torch.empty_like(single[1])

    def captured_dual():
        query, key, _ = indexer._get_q_k_bf16(
            q_lora, x, positions, True, forward_batch
        )
        static_query.copy_(query)
        static_key.copy_(key)

    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        captured_dual()
    torch.cuda.current_stream().wait_stream(warmup_stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    torch.cuda.stream = RecordingStreamContext
    RecordingStreamContext.entries.clear()
    try:
        with model_capture_mode():
            with torch.cuda.graph(graph):
                captured_dual()
        torch.cuda.synchronize()
        capture_entries = list(RecordingStreamContext.entries)
        RecordingStreamContext.entries.clear()
    finally:
        torch.cuda.stream = original_stream_context

    capture_output_equal_reference = {
        "query": compare(static_query, single[0]),
        "key": compare(static_key, single[1]),
    }
    x2 = torch.randn_like(x)
    q2 = torch.randn_like(q_lora)
    x.copy_(x2)
    q_lora.copy_(q2)
    graph.replay()
    torch.cuda.synchronize()
    replay_entries = list(RecordingStreamContext.entries)
    RecordingStreamContext.entries.clear()
    single2 = indexer._get_q_k_bf16(
        q_lora, x, positions, False, forward_batch
    )
    torch.cuda.synchronize()
    graph_result = {
        "capture_alt_stream_entries": len(capture_entries),
        "capture_output_equal_reference": capture_output_equal_reference,
        "replay_alt_stream_entries": len(replay_entries),
        "replay_query_compare": compare(static_query, single2[0]),
        "replay_key_compare": compare(static_key, single2[1]),
    }

result = {
    "runtime_option": os.environ.get("SGLANG_ROCM_USE_MULTI_STREAM", ""),
    "device": torch.cuda.get_device_name(0),
    "capability": list(torch.cuda.get_device_capability(0)),
    "torch": torch.__version__,
    "hip": torch.version.hip,
    "tokens": tokens,
    "threshold": DUAL_STREAM_TOKEN_THRESHOLD,
    "model_alt_stream_is_none": model.alt_stream is None,
    "indexer_alt_stream_is_none": indexer.alt_stream is None,
    "compress_gate_stream_is_none": indexer.compress_gate_stream is None,
    "half_device_sm_count_missing_before_diagnostic": half_device_sm_count_missing_before_diagnostic,
    "runtime_gate": runtime_gate,
    "runtime_gate_alt_stream_entries": len(runtime_gate_entries),
    "forced_dual_available": forced_dual_available,
    "forced_dual_alt_stream_entries": len(forced_entries),
    "single_query_stats": {
        "min": float(single[0].float().min().item()),
        "max": float(single[0].float().max().item()),
        "mean": float(single[0].float().mean().item()),
    },
    "single_key_stats": {
        "min": float(single[1].float().min().item()),
        "max": float(single[1].float().max().item()),
        "mean": float(single[1].float().mean().item()),
    },
}
if dual is not None:
    result["forced_dual_query_compare"] = compare(single[0], dual[0])
    result["forced_dual_key_compare"] = compare(single[1], dual[1])
if graph_result is not None:
    result["graph_capture_replay"] = graph_result
print(json.dumps(result, indent=2))
parallel_ctx.__exit__(None, None, None)
