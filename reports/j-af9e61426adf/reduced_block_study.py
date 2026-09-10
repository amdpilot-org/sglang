#!/usr/bin/env python3
"""Bounded MI300X eager versus full decode-graph replay study."""

import argparse
import json
import statistics
import subprocess
import time
from array import array
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import LlamaConfig

from sglang.benchmark.one_batch import (
    _maybe_prepare_mlp_sync_batch,
    extend,
    prepare_synthetic_inputs_for_latency_test,
)
from sglang.srt.arg_groups.overrides import resolving_view
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.distributed.parallel_state_wrapper import ParallelState
from sglang.srt.entrypoints.engine import _set_envs_and_config
from sglang.srt.layers.dp_attention import compute_dp_attention_world_info
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.runtime_context import publish
from sglang.srt.server_args import PortArgs, ServerArgs


CONFIG = {
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "hidden_size": 512,
    "intermediate_size": 1024,
    "num_hidden_layers": 2,
    "num_attention_heads": 8,
    "num_key_value_heads": 2,
    "vocab_size": 512,
    "max_position_embeddings": 128,
    "rms_norm_eps": 1e-5,
    "rope_theta": 10000.0,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
}

BATCH_SIZES = (1, 2, 4)
INPUT_LEN = 32
WARMUP_FORWARDS = 3
MEASURED_FORWARDS = 20
WEIGHT_LIMIT_BYTES = 4 * 1024**3
LIVE_ALLOCATION_LIMIT_BYTES = 48 * 1024**3
ACCURACY_GATE = {
    "cosine_min": 0.999,
    "mean_abs_max": 0.02,
    "max_abs_max": 0.10,
}


def write_model_config(model_dir: Path) -> None:
    config = LlamaConfig(**CONFIG)
    config.save_pretrained(model_dir)


def build_runner(model_dir: Path) -> ModelRunner:
    server_args = ServerArgs(
        model_path=str(model_dir),
        tokenizer_path=str(model_dir),
        load_format="dummy",
        dtype="bfloat16",
        attention_backend="triton",
        mem_fraction_static=0.05,
        max_running_requests=8,
        max_total_tokens=4096,
        context_length=128,
        disable_radix_cache=True,
        disable_prefill_cuda_graph=True,
        cuda_graph_backend_decode="full",
        cuda_graph_bs_decode=list(BATCH_SIZES),
        port=30000,
    )
    server_args.resolve_once()
    _set_envs_and_config(server_args)
    publish(server_args, role="scheduler")

    cfg = resolving_view(server_args)
    tp_rank = 0
    attn_tp_rank, attn_tp_size, attn_dp_rank, attn_dp_size = (
        compute_dp_attention_world_info(
            cfg.enable_dp_attention,
            tp_rank,
            cfg.tp_size,
            cfg.dp_size,
            cfg.attn_cp_size,
        )
    )
    parallel_state = ParallelState(
        tp_rank=tp_rank,
        tp_size=cfg.tp_size,
        pp_rank=0,
        pp_size=1,
        dp_rank=None,
        dp_size=cfg.dp_size,
        attn_tp_rank=attn_tp_rank,
        attn_tp_size=attn_tp_size,
        attn_cp_rank=0,
        attn_cp_size=cfg.attn_cp_size,
        attn_dcp_rank=tp_rank % cfg.dcp_size,
        attn_dcp_size=cfg.dcp_size,
        attn_dp_rank=attn_dp_rank,
        attn_dp_size=attn_dp_size,
        moe_ep_rank=0,
        moe_ep_size=cfg.ep_size,
        moe_dp_rank=None,
        moe_dp_size=cfg.moe_dp_size,
        gpu_id=0,
    )
    model_config = ModelConfig.from_server_args(server_args)
    runner = ModelRunner(
        model_config=model_config,
        mem_fraction_static=cfg.mem_fraction_static,
        gpu_id=0,
        ps=parallel_state,
        nccl_port=PortArgs.init_new(server_args).nccl_port,
        server_args=server_args,
    )
    runner.alloc_memory_pool()
    runner.init_attention_backends()
    runner.init_cuda_graphs()
    return runner


def input_ids_for_case(batch_size: int, case_index: int) -> list[list[int]]:
    base = 17 + case_index * 97
    return [
        [(base + row * 31 + column) % CONFIG["vocab_size"] for column in range(INPUT_LEN)]
        for row in range(batch_size)
    ]


def decode_token_ids(batch_size: int, step: int, case_index: int) -> list[int]:
    base = 113 + case_index * 53
    return [
        (base + step * batch_size + row) % CONFIG["vocab_size"]
        for row in range(batch_size)
    ]


def decode_forward(
    runner: ModelRunner, batch: Any, token_ids: list[int]
) -> tuple[Any, ForwardBatch, bool]:
    torch.cuda.synchronize()
    start = time.perf_counter()
    batch.input_ids = torch.tensor(
        token_ids, dtype=torch.int64, device=runner.device
    )
    batch.prepare_for_decode()
    _maybe_prepare_mlp_sync_batch(batch, runner)
    forward_batch = ForwardBatch.init_new(
        batch,
        runner,
        return_hidden_states_before_norm=False,
    )
    output = runner.forward(forward_batch)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return output, forward_batch, elapsed


def graph_buffer_pointers(runner: ModelRunner) -> dict[str, int]:
    buffers = runner.decode_cuda_graph_runner.buffers
    pointers = {}
    for name in (
        "input_ids",
        "positions",
        "seq_lens",
        "seq_lens_cpu",
        "out_cache_loc",
        "req_pool_indices",
    ):
        value = getattr(buffers, name, None)
        if isinstance(value, torch.Tensor):
            pointers[name] = value.data_ptr()
    return pointers


def rotate_half(tensor: torch.Tensor) -> torch.Tensor:
    first, second = tensor.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def apply_rope(
    tensor: torch.Tensor, positions: torch.Tensor, theta: float
) -> torch.Tensor:
    head_dim = tensor.shape[-1]
    exponent = torch.arange(0, head_dim, 2, device=tensor.device, dtype=torch.float32)
    inv_freq = 1.0 / (theta ** (exponent / head_dim))
    frequencies = positions.float()[..., None] * inv_freq[None, :]
    embedding = torch.cat((frequencies, frequencies), dim=-1)
    embedding = embedding.unsqueeze(2)
    return tensor * embedding.cos() + rotate_half(tensor) * embedding.sin()


def rms_norm(tensor: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    variance = tensor.float().pow(2).mean(dim=-1, keepdim=True)
    return tensor.float() * torch.rsqrt(variance + eps) * weight.float()


def independent_torch_logits(model: torch.nn.Module, token_ids: list[list[int]]) -> torch.Tensor:
    config = model.config
    rope_parameters = getattr(config, "rope_parameters", None)
    rope_theta = (
        rope_parameters.get("rope_theta", 10000.0)
        if rope_parameters is not None
        else getattr(config, "rope_theta", 10000.0)
    )
    device = model.model.embed_tokens.weight.device
    batch_size = len(token_ids)
    sequence_length = len(token_ids[0])
    tokens = torch.tensor(token_ids, dtype=torch.int64, device=device)
    positions = torch.arange(sequence_length, device=device).repeat(batch_size, 1)
    hidden = F.embedding(tokens, model.model.embed_tokens.weight.float())

    for layer in model.model.layers:
        residual = hidden
        hidden = rms_norm(hidden, layer.input_layernorm.weight, config.rms_norm_eps)
        qkv = F.linear(hidden, layer.self_attn.qkv_proj.weight.float())
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        q_size = config.num_attention_heads * head_dim
        kv_size = config.num_key_value_heads * head_dim
        query, key, value = qkv.split((q_size, kv_size, kv_size), dim=-1)
        query = query.view(
            batch_size,
            sequence_length,
            config.num_attention_heads,
            -1,
        )
        key = key.view(
            batch_size,
            sequence_length,
            config.num_key_value_heads,
            -1,
        )
        value = value.view(
            batch_size,
            sequence_length,
            config.num_key_value_heads,
            -1,
        )
        query = apply_rope(query, positions, rope_theta)
        key = apply_rope(key, positions, rope_theta)
        repeat_count = config.num_attention_heads // config.num_key_value_heads
        key = key.repeat_interleave(repeat_count, dim=2)
        value = value.repeat_interleave(repeat_count, dim=2)
        attention = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            is_causal=True,
        )
        attention = attention.transpose(1, 2).reshape(
            batch_size, sequence_length, -1
        )
        hidden = F.linear(attention, layer.self_attn.o_proj.weight.float())
        hidden = residual + hidden

        residual = hidden
        hidden = rms_norm(
            hidden, layer.post_attention_layernorm.weight, config.rms_norm_eps
        )
        gate_up = F.linear(hidden, layer.mlp.gate_up_proj.weight.float())
        gate, up = gate_up.chunk(2, dim=-1)
        hidden = F.linear(
            F.silu(gate) * up,
            layer.mlp.down_proj.weight.float(),
        )
        hidden = residual + hidden

    hidden = rms_norm(hidden, model.model.norm.weight, config.rms_norm_eps)
    logits = F.linear(hidden[:, -1], model.lm_head.weight.float())
    return logits


def accuracy_metrics(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    actual = actual.float().cpu()
    reference = reference.float().cpu()
    difference = (actual - reference).abs()
    cosine = F.cosine_similarity(
        actual.flatten(), reference.flatten(), dim=0
    ).item()
    return {
        "cosine": cosine,
        "mean_abs": difference.mean().item(),
        "max_abs": difference.max().item(),
    }


def passes_gate(metrics: dict[str, float]) -> bool:
    return (
        metrics["cosine"] >= ACCURACY_GATE["cosine_min"]
        and metrics["mean_abs"] <= ACCURACY_GATE["mean_abs_max"]
        and metrics["max_abs"] <= ACCURACY_GATE["max_abs_max"]
    )


def run_case(
    runner: ModelRunner,
    batch_size: int,
    case_index: int,
    mode: str,
) -> tuple[dict[str, Any], torch.Tensor]:
    runner.req_to_token_pool.clear()
    runner.token_to_kv_pool_allocator.clear()
    graph_runner = runner.decode_cuda_graph_runner
    runner.decode_cuda_graph_runner = None if mode == "eager" else graph_runner

    input_ids = input_ids_for_case(batch_size, case_index)
    reqs = prepare_synthetic_inputs_for_latency_test(
        batch_size,
        INPUT_LEN,
        input_ids,
    )
    _, _, batch = extend(reqs, runner)
    sequence = [list(row) for row in input_ids]
    latencies = []
    graph_admissions = []
    fresh_inputs = []
    buffer_pointers = []
    final_logits = None
    previous_logits = None

    for step in range(WARMUP_FORWARDS + MEASURED_FORWARDS):
        token_ids = decode_token_ids(batch_size, step, case_index)
        fresh_inputs.append(token_ids)
        for row, token in enumerate(token_ids):
            sequence[row].append(token)
        output, _, elapsed = decode_forward(runner, batch, token_ids)
        logits = output.logits_output.next_token_logits
        graph_admissions.append(bool(output.can_run_graph))
        if mode == "replay":
            buffer_pointers.append(graph_buffer_pointers(runner))
        if step >= WARMUP_FORWARDS:
            latencies.append(elapsed)
        previous_logits = final_logits
        final_logits = logits.detach().clone()

    runner.decode_cuda_graph_runner = graph_runner
    reference = independent_torch_logits(runner.model, sequence)
    metrics = accuracy_metrics(final_logits, reference)
    output_change = (
        (final_logits - previous_logits).abs().max().item()
        if previous_logits is not None
        else None
    )
    pointer_sets = {
        name: sorted({pointers[name] for pointers in buffer_pointers})
        for name in (buffer_pointers[0] if buffer_pointers else {})
    }

    result = {
        "mode": mode,
        "batch_size": batch_size,
        "input_len": INPUT_LEN,
        "decode_forwards": WARMUP_FORWARDS + MEASURED_FORWARDS,
        "warmup_forwards": WARMUP_FORWARDS,
        "measured_forwards": MEASURED_FORWARDS,
        "latency_seconds": {
            "mean": statistics.fmean(latencies),
            "median": statistics.median(latencies),
            "min": min(latencies),
            "max": max(latencies),
            "values": latencies,
        },
        "graph_admissions": graph_admissions,
        "fresh_input_first_step": fresh_inputs[0],
        "fresh_input_last_step": fresh_inputs[-1],
        "static_buffer_data_ptrs": pointer_sets,
        "accuracy_vs_independent_torch": metrics,
        "accuracy_gate_passed": passes_gate(metrics),
        "last_output_change_abs": output_change,
        "final_logits_sha256": __import__("hashlib").sha256(
            final_logits.detach().cpu().numpy().tobytes()
        ).hexdigest(),
    }
    return result, final_logits


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    started_at = time.time()
    torch.manual_seed(20260910)
    cache_dir = Path("/tmp/sglang-cache-j-af9e61426adf")
    model_dir = cache_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    write_model_config(model_dir)

    runner = build_runner(model_dir)
    weight_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in runner.model.parameters()
    )
    torch.cuda.reset_peak_memory_stats()
    initial_allocated = torch.cuda.memory_allocated()

    cases = []
    final_logits_by_batch = {}
    for batch_size in BATCH_SIZES:
        for mode in ("eager", "replay"):
            case, final_logits = run_case(runner, batch_size, batch_size, mode)
            cases.append(case)
            if mode == "eager":
                final_logits_by_batch[batch_size] = final_logits
            else:
                eager_logits = final_logits_by_batch[batch_size]
                cases[-1]["eager_vs_replay_max_abs"] = (
                    (final_logits - eager_logits).abs().max().item()
                )

    peak_allocated = torch.cuda.max_memory_allocated()
    result = {
        "label": "reduced real SGLang ModelRunner study",
        "started_at_unix": int(started_at),
        "elapsed_seconds": time.time() - started_at,
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "torch_hip_version": torch.version.hip,
        },
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "model_config": CONFIG,
        "weight_generation": {
            "loader": "SGLang DummyModelLoader",
            "seed": 20260910,
            "dtype": "torch.bfloat16",
            "parameter_bytes": weight_bytes,
            "limit_bytes": WEIGHT_LIMIT_BYTES,
            "within_limit": weight_bytes < WEIGHT_LIMIT_BYTES,
        },
        "execution": {
            "engine": "sglang.srt.model_executor.model_runner.ModelRunner",
            "attention_backend": "triton",
            "decode_graph_backend": "full",
            "capture_batch_sizes": list(BATCH_SIZES),
            "timing_method": "time.perf_counter around decode batch preparation, ModelRunner.forward, and torch.cuda.synchronize",
            "accuracy_gate": ACCURACY_GATE,
            "independent_control": "standalone PyTorch Llama forward using copied SGLang weights",
        },
        "unsupported_capture_boundaries": {
            "batch_size_above_max": "batch_size > max(capture_batch_sizes)=4 falls back to eager",
            "token_embedding_overrides": "replace_embeds is not None falls back to eager",
            "speculative_width_mismatch": "spec_info.num_tokens_per_req != captured_req_width falls back to eager",
            "mixed_encoder_batch": "encoder-decoder mixed batch with encoder_lens == 0 falls back to eager",
            "two_batch_overlap_unsupported": "forward_batch.can_run_tbo is false when TBO is enabled",
            "ngram_shape_mismatch": "ngram batch_size * captured_req_width != input_ids.numel falls back to eager",
        },
        "memory": {
            "initial_allocated_bytes": initial_allocated,
            "peak_allocated_bytes": peak_allocated,
            "reserved_bytes": torch.cuda.memory_reserved(),
            "limit_bytes": LIVE_ALLOCATION_LIMIT_BYTES,
            "within_limit": peak_allocated < LIVE_ALLOCATION_LIMIT_BYTES,
        },
        "cases": cases,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    all_gates = all(case["accuracy_gate_passed"] for case in cases)
    all_admissions = all(
        all(case["graph_admissions"])
        for case in cases
        if case["mode"] == "replay"
    )
    return 0 if all_gates and all_admissions else 1


if __name__ == "__main__":
    raise SystemExit(main())
