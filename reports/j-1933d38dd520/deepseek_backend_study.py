#!/usr/bin/env python3
"""Bounded MI300X DeepSeek V2 AITER versus Triton backend study."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import DeepseekV2Config

from aiter.mla import mla_decode_fwd
from sglang.benchmark.one_batch import (
    _maybe_prepare_mlp_sync_batch,
    extend,
    prepare_synthetic_inputs_for_latency_test,
)
from sglang.kernels.ops.attention.decode_attention import decode_attention_fwd
from sglang.srt.arg_groups.overrides import resolving_view
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.distributed.parallel_state_wrapper import ParallelState
from sglang.srt.entrypoints.engine import _set_envs_and_config
from sglang.srt.layers.dp_attention import compute_dp_attention_world_info
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import fused_experts
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.models.deepseek_v2 import DeepseekV2ForCausalLM
from sglang.srt.runtime_context import publish
from sglang.srt.server_args import PortArgs, ServerArgs


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
IMAGE_REFERENCE = "amdpilotv2/open-job-mi300:jit-config-readable-35122-260909"
IMAGE_LOCAL_ID = "sha256:dfc9419089c338b5712da4841768b38b1ab79f3da41f8c58c3cd4dfcc1147ff1"

CONFIG = DeepseekV2Config(
    architectures=["DeepseekV2ForCausalLM"],
    hidden_size=512,
    intermediate_size=512,
    moe_intermediate_size=512,
    num_hidden_layers=2,
    num_attention_heads=8,
    num_key_value_heads=8,
    vocab_size=512,
    max_position_embeddings=128,
    rms_norm_eps=1e-5,
    q_lora_rank=512,
    kv_lora_rank=512,
    qk_nope_head_dim=128,
    qk_rope_head_dim=64,
    v_head_dim=128,
    n_routed_experts=1,
    num_experts_per_tok=1,
    n_group=1,
    topk_group=1,
    scoring_func="softmax",
    n_shared_experts=1,
    first_k_dense_replace=0,
    moe_layer_freq=1,
    torch_dtype="bfloat16",
)


def write_model_config(model_dir: Path) -> None:
    CONFIG.save_pretrained(model_dir)


def build_runner(model_dir: Path, backend: str) -> ModelRunner:
    server_args = ServerArgs(
        model_path=str(model_dir),
        tokenizer_path=str(model_dir),
        load_format="dummy",
        dtype="bfloat16",
        attention_backend=backend,
        mem_fraction_static=0.05,
        max_running_requests=8,
        max_total_tokens=4096,
        context_length=128,
        disable_radix_cache=True,
        disable_cuda_graph=True,
        disable_prefill_cuda_graph=True,
        port=30000,
    )
    server_args.resolve_once()
    _set_envs_and_config(server_args)
    publish(server_args, role="scheduler")

    cfg = resolving_view(server_args)
    attn_tp_rank, attn_tp_size, attn_dp_rank, attn_dp_size = (
        compute_dp_attention_world_info(
            cfg.enable_dp_attention,
            0,
            cfg.tp_size,
            cfg.dp_size,
            cfg.attn_cp_size,
        )
    )
    parallel_state = ParallelState(
        tp_rank=0,
        tp_size=cfg.tp_size,
        pp_rank=0,
        pp_size=1,
        dp_rank=None,
        dp_size=cfg.dp_size,
        attn_tp_rank=attn_tp_rank,
        attn_tp_size=attn_tp_size,
        attn_cp_rank=0,
        attn_cp_size=cfg.attn_cp_size,
        attn_dcp_rank=0,
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
        [(base + row * 31 + column) % CONFIG.vocab_size for column in range(INPUT_LEN)]
        for row in range(batch_size)
    ]


def decode_token_ids(batch_size: int, step: int, case_index: int) -> list[int]:
    base = 113 + case_index * 53
    return [
        (base + step * batch_size + row) % CONFIG.vocab_size
        for row in range(batch_size)
    ]


def decode_forward(
    runner: ModelRunner, batch: Any, token_ids: list[int]
) -> tuple[Any, ForwardBatch, float]:
    torch.cuda.synchronize()
    start = time.perf_counter()
    batch.input_ids = torch.tensor(token_ids, dtype=torch.int64, device=runner.device)
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


def accuracy_metrics(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    actual = actual.detach().float().cpu()
    reference = reference.detach().float().cpu()
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


def attention_reference(
    runner: ModelRunner,
    q: torch.Tensor,
    forward_batch: ForwardBatch,
    scale: float,
) -> torch.Tensor:
    key_buffer = runner.token_to_kv_pool.get_key_buffer(0)
    value_buffer = runner.token_to_kv_pool.get_value_buffer(0)
    req_to_token = runner.req_to_token_pool.req_to_token
    req_indices = forward_batch.req_pool_indices.detach().cpu().tolist()
    seq_lens = forward_batch.seq_lens.detach().cpu().tolist()
    output = torch.empty_like(q[..., : value_buffer.shape[-1]])

    for row, (request_index, sequence_length) in enumerate(
        zip(req_indices, seq_lens)
    ):
        slots = req_to_token[request_index, :sequence_length].long()
        keys = key_buffer[slots, 0, :].float()
        values = value_buffer[slots, 0, :].float()
        scores = torch.einsum("hd,td->ht", q[row].float(), keys) * scale
        probabilities = torch.softmax(scores, dim=-1)
        output[row] = torch.einsum("ht,td->hd", probabilities, values).to(
            output.dtype
        )
    return output


def mlp_reference(layer: torch.nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    hidden = hidden_states.detach().float()

    shared_gate_up = F.linear(
        hidden, layer.shared_experts.gate_up_proj.weight.detach().float()
    )
    shared_gate, shared_up = shared_gate_up.chunk(2, dim=-1)
    shared_output = F.linear(
        F.silu(shared_gate) * shared_up,
        layer.shared_experts.down_proj.weight.detach().float(),
    )

    router_logits = F.linear(hidden, layer.gate.weight.detach().float())
    router_weight = torch.softmax(router_logits, dim=-1)[:, 0]
    w13 = layer.experts.w13_weight.detach()[0].float()
    w2 = layer.experts.w2_weight.detach()[0].float()
    gate_up = F.linear(hidden, w13)
    gate, up = gate_up.chunk(2, dim=-1)
    routed_output = F.linear(F.silu(gate) * up, w2)
    return shared_output + router_weight[:, None] * routed_output


def install_hooks(runner: ModelRunner) -> tuple[dict[str, Any], callable]:
    records: dict[str, Any] = {}
    layer = runner.model.model.layers[0]
    original_attention = layer.self_attn.attn_mqa.forward
    original_mlp = layer.mlp.forward

    def attention_forward(q, k, v, forward_batch, *args, **kwargs):
        output = original_attention(q, k, v, forward_batch, *args, **kwargs)
        records["attention"] = {
            "q": q.detach().clone(),
            "output": output.detach().clone(),
            "forward_batch": forward_batch,
            "scale": layer.self_attn.scaling,
        }
        return output

    def mlp_forward(hidden_states, *args, **kwargs):
        input_copy = hidden_states.detach().clone()
        output = original_mlp(hidden_states, *args, **kwargs)
        records["mlp"] = {
            "input": input_copy,
            "output": output.detach().clone(),
            "layer": layer.mlp,
        }
        return output

    layer.self_attn.attn_mqa.forward = attention_forward
    layer.mlp.forward = mlp_forward

    def cleanup() -> None:
        layer.self_attn.attn_mqa.forward = original_attention
        layer.mlp.forward = original_mlp

    return records, cleanup


def run_case(
    runner: ModelRunner,
    backend: str,
    batch_size: int,
    case_index: int,
) -> dict[str, Any]:
    runner.req_to_token_pool.clear()
    runner.token_to_kv_pool_allocator.clear()

    input_ids = input_ids_for_case(batch_size, case_index)
    requests = prepare_synthetic_inputs_for_latency_test(
        batch_size,
        INPUT_LEN,
        input_ids,
    )
    _, _, batch = extend(requests, runner)

    latencies = []
    final_logits = None
    for step in range(WARMUP_FORWARDS + MEASURED_FORWARDS):
        token_ids = decode_token_ids(batch_size, step, case_index)
        output, _, elapsed = decode_forward(runner, batch, token_ids)
        if step >= WARMUP_FORWARDS:
            latencies.append(elapsed)
        final_logits = output.logits_output.next_token_logits.detach().clone()

    records, cleanup = install_hooks(runner)
    validation_token_ids = decode_token_ids(
        batch_size, WARMUP_FORWARDS + MEASURED_FORWARDS, case_index
    )
    output, _, _ = decode_forward(runner, batch, validation_token_ids)
    cleanup()

    attention_record = records["attention"]
    attention_expected = attention_reference(
        runner,
        attention_record["q"],
        attention_record["forward_batch"],
        attention_record["scale"],
    )
    attention_actual = attention_record["output"].view(
        -1,
        attention_record["q"].shape[1],
        runner.token_to_kv_pool.get_value_buffer(0).shape[-1],
    )
    attention_metrics = accuracy_metrics(
        attention_actual, attention_expected
    )

    mlp_record = records["mlp"]
    mlp_expected = mlp_reference(mlp_record["layer"], mlp_record["input"])
    mlp_metrics = accuracy_metrics(mlp_record["output"], mlp_expected)
    if os.environ.get("DEBUG_MLP"):
        print("MLP_DEBUG")
        print("input", mlp_record["input"].shape, mlp_record["input"][0, :4])
        print("actual", mlp_record["output"].shape, mlp_record["output"][0, :8])
        print("expected", mlp_expected.shape, mlp_expected[0, :8])
        print("w13", mlp_record["layer"].experts.w13_weight.shape)
        print("w2", mlp_record["layer"].experts.w2_weight.shape)
        print("gate", mlp_record["layer"].gate.weight.shape)

    validation_logits = output.logits_output.next_token_logits.detach().clone()
    return {
        "backend": backend,
        "batch_size": batch_size,
        "input_len": INPUT_LEN,
        "decode_forwards": WARMUP_FORWARDS + MEASURED_FORWARDS + 1,
        "warmup_forwards": WARMUP_FORWARDS,
        "measured_forwards": MEASURED_FORWARDS,
        "validation_forwards": 1,
        "latency_seconds": {
            "mean": statistics.fmean(latencies),
            "median": statistics.median(latencies),
            "min": min(latencies),
            "max": max(latencies),
            "values": latencies,
        },
        "attention_reference_metrics": attention_metrics,
        "attention_gate_passed": passes_gate(attention_metrics),
        "mlp_reference_metrics": mlp_metrics,
        "mlp_gate_passed": passes_gate(mlp_metrics),
        "final_logits": validation_logits.detach().cpu().tolist(),
        "final_logits_sha256": hashlib.sha256(
            validation_logits.detach().cpu().numpy().tobytes()
        ).hexdigest(),
    }


def weight_fingerprint(model: torch.nn.Module) -> tuple[str, int]:
    digest = hashlib.sha256()
    weight_bytes = 0
    for name, parameter in sorted(model.named_parameters()):
        weight_bytes += parameter.numel() * parameter.element_size()
        digest.update(name.encode())
        digest.update(str(tuple(parameter.shape)).encode())
        digest.update(str(parameter.dtype).encode())
        digest.update(
            parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        )
    return digest.hexdigest(), weight_bytes


def native_paths(runner: ModelRunner, backend: str) -> dict[str, str]:
    paths = {
        "python": "/opt/venv/bin/python",
        "torch": inspect.getfile(torch),
        "sglang_model": inspect.getfile(DeepseekV2ForCausalLM),
        "attention_backend": inspect.getfile(type(runner.attn_backend)),
        "moe_kernel": inspect.getfile(fused_experts),
    }
    if backend == "aiter":
        paths.update(
            {
                "attention_decode_wrapper": inspect.getfile(mla_decode_fwd),
                "attention_decode_native_op": "torch.ops.aiter.mla_decode_stage1_asm_fwd",
                "attention_decode_observed_hsaco": "/sgl-workspace/aiter/hsa//gfx942/mla/mla_dec_stage1_bf16_a16w16_subQ16_mqa16.co",
                "aiter_native_core": "/sgl-workspace/aiter/aiter/jit/module_aiter_core.so",
            }
        )
    else:
        paths.update(
            {
                "attention_decode_wrapper": inspect.getfile(decode_attention_fwd),
                "attention_decode_kernel": "sglang.kernels.ops.attention.decode_attention._fwd_grouped_kernel_stage1",
            }
        )
    return paths


def gpu_identity() -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(0)
    identity = {
        "name": properties.name,
        "capability": list(torch.cuda.get_device_capability(0)),
        "total_memory_bytes": properties.total_memory,
        "torch_hip_version": torch.version.hip,
    }
    try:
        identity["rocm_smi"] = subprocess.check_output(
            [
                "rocm-smi",
                "--showproductname",
                "--showdriverversion",
                "--showserial",
                "--showuniqueid",
            ],
            text=True,
            timeout=10,
        ).strip()
    except Exception as error:
        identity["rocm_smi_error"] = repr(error)
    return identity


def run_backend(backend: str, output: Path) -> None:
    started_at = time.time()
    torch.manual_seed(20260910)
    cache_dir = Path("/tmp/sglang-cache-j-1933d38dd520")
    model_dir = cache_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    write_model_config(model_dir)

    runner = build_runner(model_dir, backend)
    for parameter in runner.model.parameters():
        if torch.is_floating_point(parameter):
            parameter.data.mul_(100.0)
    fingerprint, weight_bytes = weight_fingerprint(runner.model)
    torch.cuda.reset_peak_memory_stats()
    initial_allocated = torch.cuda.memory_allocated()

    cases = [
        run_case(runner, backend, batch_size, batch_size)
        for batch_size in BATCH_SIZES
    ]
    peak_allocated = torch.cuda.max_memory_allocated()
    result = {
        "label": "bounded real SGLang DeepSeek V2 backend comparison",
        "backend": backend,
        "started_at_unix": int(started_at),
        "elapsed_seconds": time.time() - started_at,
        "gpu": gpu_identity(),
        "image": {
            "reference": IMAGE_REFERENCE,
            "local_id": IMAGE_LOCAL_ID,
        },
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "model_config": CONFIG.to_dict(),
        "weight_generation": {
            "loader": "SGLang DummyModelLoader followed by deterministic x100 parameter scale",
            "seed": 1234,
            "dtype": "torch.bfloat16",
            "parameter_bytes": weight_bytes,
            "limit_bytes": WEIGHT_LIMIT_BYTES,
            "within_limit": weight_bytes < WEIGHT_LIMIT_BYTES,
            "fingerprint_sha256": fingerprint,
        },
        "execution": {
            "engine": "sglang.srt.model_executor.model_runner.ModelRunner",
            "attention_backend": backend,
            "cuda_graphs": "disabled",
            "timing_method": "time.perf_counter around decode batch preparation, ModelRunner.forward, and torch.cuda.synchronize",
            "accuracy_gate": ACCURACY_GATE,
            "independent_references": {
                "attention": "Torch einsum/softmax over captured MLA query and KV cache",
                "mlp": "Torch linear/SiLU formula over captured MoE input and copied weights",
            },
        },
        "native_paths": native_paths(runner, backend),
        "memory": {
            "initial_allocated_bytes": initial_allocated,
            "peak_allocated_bytes": peak_allocated,
            "reserved_bytes": torch.cuda.memory_reserved(),
            "limit_bytes": LIVE_ALLOCATION_LIMIT_BYTES,
            "within_limit": peak_allocated < LIVE_ALLOCATION_LIMIT_BYTES,
        },
        "cases": cases,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def combine_results(aiter_path: Path, triton_path: Path, output: Path) -> None:
    aiter_result = json.loads(aiter_path.read_text())
    triton_result = json.loads(triton_path.read_text())
    if aiter_result["weight_generation"]["fingerprint_sha256"] != triton_result[
        "weight_generation"
    ]["fingerprint_sha256"]:
        raise RuntimeError("AITER and Triton weight fingerprints differ")

    comparisons = []
    for aiter_case, triton_case in zip(
        aiter_result["cases"], triton_result["cases"]
    ):
        if aiter_case["batch_size"] != triton_case["batch_size"]:
            raise RuntimeError("case batch sizes differ")
        aiter_logits = torch.tensor(aiter_case["final_logits"])
        triton_logits = torch.tensor(triton_case["final_logits"])
        comparisons.append(
            {
                "batch_size": aiter_case["batch_size"],
                "aiter_mean_latency_seconds": aiter_case["latency_seconds"]["mean"],
                "triton_mean_latency_seconds": triton_case["latency_seconds"]["mean"],
                "aiter_vs_triton": accuracy_metrics(aiter_logits, triton_logits),
                "aiter_speedup_vs_triton": (
                    triton_case["latency_seconds"]["mean"]
                    / aiter_case["latency_seconds"]["mean"]
                ),
            }
        )

    result = {
        "label": "bounded MI300X DeepSeek V2 AITER versus Triton comparison",
        "backends": ["aiter", "triton"],
        "weight_fingerprint_sha256": aiter_result["weight_generation"][
            "fingerprint_sha256"
        ],
        "cases": comparisons,
        "aiter_result": aiter_result,
        "triton_result": triton_result,
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("aiter", "triton"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--combine", action="store_true")
    parser.add_argument("--aiter-result", type=Path)
    parser.add_argument("--triton-result", type=Path)
    args = parser.parse_args()

    if args.combine:
        if args.aiter_result is None or args.triton_result is None or args.output is None:
            parser.error("--combine requires --aiter-result, --triton-result, and --output")
        combine_results(args.aiter_result, args.triton_result, args.output)
        return 0
    if args.backend is None or args.output is None:
        parser.error("a backend run requires --backend and --output")
    run_backend(args.backend, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
