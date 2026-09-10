"""Bounded synthetic-model memory-scaling benchmark for one GPU.

This benchmark uses a locally generated tiny Llama configuration and
``DummyModelLoader`` random weights.  It performs real SGLang
``ModelRunner.extend`` and ``ModelRunner.decode`` calls, compares each greedy
token with an independent Torch Llama implementation, and records predicted
activation/KV workspace bytes beside CUDA live/peak allocations.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
from array import array
from pathlib import Path
from typing import Any, Dict, List, Tuple

os.environ.setdefault("USE_ROCM_AITER_ROPE_BACKEND", "0")

import numpy as np
import torch
import torch.nn.functional as F
from transformers import LlamaConfig

from sglang.srt.arg_groups.overrides import resolving_view
from sglang.srt.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
)
from sglang.srt.entrypoints.engine import _set_envs_and_config
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.runtime_context import publish
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import configure_logger

from sglang.benchmark.one_batch import decode, extend, load_model


class TorchRMSNorm(torch.nn.Module):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return (hidden_states * self.weight.float()).to(input_dtype)


class TorchLlamaAttention(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rope_theta: float,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        self.q_size = num_heads * head_dim
        self.kv_size = num_kv_heads * head_dim
        self.qkv_proj = torch.nn.Linear(hidden_size, self.q_size + 2 * self.kv_size, bias=False)
        self.o_proj = torch.nn.Linear(self.q_size, hidden_size, bias=False)

    def _rope(self, tensor: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        inverse_frequency = 1.0 / (
            self.rope_theta
            ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=tensor.device) / self.head_dim)
        )
        frequencies = positions.float().unsqueeze(-1) * inverse_frequency
        cosine = frequencies.cos().to(tensor.dtype).unsqueeze(-2)
        sine = frequencies.sin().to(tensor.dtype).unsqueeze(-2)
        first, second = tensor.chunk(2, dim=-1)
        return torch.cat(
            (first * cosine - second * sine, second * cosine + first * sine),
            dim=-1,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        batch_size: int,
        context_length: int,
    ) -> torch.Tensor:
        fused = self.qkv_proj(hidden_states)
        query, key, value = fused.split((self.q_size, self.kv_size, self.kv_size), dim=-1)
        query = query.view(batch_size, context_length, self.num_heads, self.head_dim)
        key = key.view(batch_size, context_length, self.num_kv_heads, self.head_dim)
        value = value.view(batch_size, context_length, self.num_kv_heads, self.head_dim)
        query = self._rope(query, positions)
        key = self._rope(key, positions)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        if self.num_heads != self.num_kv_heads:
            repeat_count = self.num_heads // self.num_kv_heads
            key = key.repeat_interleave(repeat_count, dim=1)
            value = value.repeat_interleave(repeat_count, dim=1)
        attention = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
        )
        attention = attention.transpose(1, 2).reshape(
            batch_size, context_length, self.q_size
        )
        return self.o_proj(attention)


class TorchLlamaMLP(torch.nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_up_proj = torch.nn.Linear(hidden_size, 2 * intermediate_size, bias=False)
        self.down_proj = torch.nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(hidden_states).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class TorchLlamaLayer(torch.nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        head_dim = config.hidden_size // config.num_attention_heads
        rope_parameters = getattr(config, "rope_parameters", None) or {}
        rope_theta = rope_parameters.get(
            "rope_theta", getattr(config, "rope_theta", 10000.0)
        )
        self.self_attn = TorchLlamaAttention(
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            head_dim,
            rope_theta,
        )
        self.mlp = TorchLlamaMLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = TorchRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = TorchRMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        batch_size: int,
        context_length: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, positions, batch_size, context_length)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class TorchLlamaModel(torch.nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = torch.nn.ModuleList([TorchLlamaLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = TorchRMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        batch_size: int,
        context_length: int,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, positions, batch_size, context_length)
        return self.norm(hidden_states + residual)


class TorchLlamaForCausalLM(torch.nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.model = TorchLlamaModel(config)
        self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        batch_size: int,
        context_length: int,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, batch_size, context_length)
        return self.lm_head(hidden_states)


def create_local_config(config_dir: Path, seed: int) -> LlamaConfig:
    config_dir.mkdir(parents=True, exist_ok=True)
    config = LlamaConfig(
        vocab_size=512,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=1024,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        tie_word_embeddings=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    config.architectures = ["LlamaForCausalLM"]
    config.dtype = torch.bfloat16
    config.save_pretrained(config_dir)
    generation_config = {
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
        "do_sample": False,
        "max_length": 1024,
        "transformers_version": "local-synthetic",
    }
    with (config_dir / "generation_config.json").open("w") as output:
        json.dump(generation_config, output, indent=2)
    with (config_dir / "synthetic_seed.txt").open("w") as output:
        output.write(str(seed))
    return config


def make_requests(
    batch_size: int,
    context_length: int,
    vocab_size: int,
    output_length: int,
    seed: int,
) -> Tuple[np.ndarray, List[Req]]:
    generator = np.random.default_rng(seed)
    input_ids = generator.integers(0, vocab_size, (batch_size, context_length), dtype=np.int32)
    sampling_params = SamplingParams(temperature=0, max_new_tokens=output_length)
    requests = []
    for request_index in range(batch_size):
        request = Req(
            rid=request_index,
            origin_input_text="",
            origin_input_ids=array("q", input_ids[request_index]),
            sampling_params=sampling_params,
        )
        request.full_untruncated_fill_ids = request.origin_input_ids
        request.logprob_start_len = -1
        request.set_extend_range(0, context_length)
        requests.append(request)
    return input_ids, requests


def tensor_bytes(value: Any) -> int:
    if isinstance(value, list):
        return sum(tensor_bytes(item) for item in value)
    return int(value.numel() * value.element_size())


def state_dict_bytes(state_dict: Dict[str, torch.Tensor]) -> int:
    return sum(tensor_bytes(value) for value in state_dict.values())


def predicted_activation_bytes(
    config: LlamaConfig,
    batch_size: int,
    context_length: int,
    phase: str,
) -> int:
    item_size = torch.bfloat16.itemsize
    token_count = batch_size * (context_length if phase == "prefill" else 1)
    attention_rows = batch_size * (context_length * context_length if phase == "prefill" else context_length)
    head_dim = config.hidden_size // config.num_attention_heads
    query_size = config.num_attention_heads * head_dim
    kv_size = config.num_key_value_heads * head_dim
    per_layer = (
        token_count * config.hidden_size * 4
        + token_count * (query_size + 2 * kv_size) * 2
        + attention_rows * config.num_attention_heads * item_size
        + token_size(token_count, query_size, item_size)
        + token_count * config.intermediate_size * 4
    )
    return (
        token_count * 8 * 2
        + config.num_hidden_layers * per_layer
        + token_count * config.hidden_size * item_size
        + batch_size * config.vocab_size * item_size
    )


def token_size(token_count: int, feature_size: int, item_size: int) -> int:
    return token_count * feature_size * item_size


def timed_forward(function, *arguments):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    pre_allocated = torch.cuda.memory_allocated()
    pre_reserved = torch.cuda.memory_reserved()
    start = time.perf_counter()
    result = function(*arguments)
    torch.cuda.synchronize()
    latency = time.perf_counter() - start
    return {
        "result": result,
        "latency_seconds": latency,
        "pre_live_allocated_bytes": pre_allocated,
        "pre_live_reserved_bytes": pre_reserved,
        "post_live_allocated_bytes": torch.cuda.memory_allocated(),
        "post_live_reserved_bytes": torch.cuda.memory_reserved(),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_allocated_delta_bytes": torch.cuda.max_memory_allocated() - pre_allocated,
    }


def control_logits(
    control: TorchLlamaForCausalLM,
    sequences: List[List[int]],
    device: torch.device,
) -> torch.Tensor:
    lengths = [len(sequence) for sequence in sequences]
    if len(set(lengths)) != 1:
        raise ValueError(f"Control requires equal-length sequences, got {lengths}")
    context_length = lengths[0]
    batch_size = len(sequences)
    input_ids = torch.tensor(sequences, dtype=torch.int64, device=device)
    positions = torch.arange(context_length, dtype=torch.int64, device=device)
    positions = positions.unsqueeze(0).expand(batch_size, -1)
    logits = control(input_ids, positions, batch_size, context_length)
    return logits.view(batch_size, context_length, -1)[:, -1, :]


def module_path(name: str) -> str | None:
    spec = importlib.util.find_spec(name)
    return getattr(spec, "origin", None) if spec else None


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.tolist()
    return str(value)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("synthetic-memory-scaling.json"))
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--context-lengths", type=int, nargs="+", default=[128, 512])
    parser.add_argument("--output-length", type=int, default=8)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--logit-atol", type=float, default=0.10)
    parser.add_argument("--logit-rtol", type=float, default=0.10)
    return parser.parse_args()


def main():
    arguments = parse_args()
    if len(arguments.batch_sizes) * len(arguments.context_lengths) > 6:
        raise ValueError("This benchmark supports at most six workload cases")
    if torch.cuda.device_count() != 1:
        raise ValueError(f"Expected exactly one GPU, got {torch.cuda.device_count()}")

    torch.manual_seed(arguments.seed)
    torch.cuda.manual_seed_all(arguments.seed)
    torch.set_num_threads(1)
    config_dir = Path(tempfile.mkdtemp(prefix="sglang-synthetic-memory-"))
    config = create_local_config(config_dir, arguments.seed)

    server_args = ServerArgs(
        model_path=str(config_dir),
        load_format="dummy",
        dtype="bfloat16",
        attention_backend="triton",
        disable_cuda_graph=True,
        mem_fraction_static=0.02,
        max_total_tokens=4096,
        max_running_requests=16,
        skip_tokenizer_init=True,
    )
    server_args.resolve_once()
    _set_envs_and_config(server_args)
    publish(server_args, role="scheduler")
    configure_logger(server_args, prefix=" TP0")
    port_args = PortArgs.init_new(server_args)
    bench_runner, _ = load_model(
        server_args,
        port_args,
        gpu_id=0,
        tp_rank=0,
        load_tokenizer=False,
    )
    runner = bench_runner.torch_runner

    device = torch.device("cuda:0")
    control = TorchLlamaForCausalLM(config).to(device=device, dtype=torch.bfloat16)
    control.load_state_dict(runner.model.state_dict())
    control.eval()

    kv_pool = runner.token_to_kv_pool_allocator.get_kvcache()
    kv_size = kv_pool.get_kv_size_bytes()
    predicted_kv_bytes = sum(kv_size)
    request_pool_bytes = tensor_bytes(runner.req_to_token_pool.req_to_token)
    weights_bytes = state_dict_bytes(runner.model.state_dict())
    control_weights_bytes = state_dict_bytes(control.state_dict())

    results = []
    for batch_size in arguments.batch_sizes:
        for context_length in arguments.context_lengths:
            case_seed = arguments.seed + batch_size * 1009 + context_length
            case = {
                "batch_size": batch_size,
                "context_length": context_length,
                "output_length": arguments.output_length,
                "seed": case_seed,
                "repetitions": [],
            }
            for repetition in range(arguments.warmups + arguments.repetitions):
                measured = repetition >= arguments.warmups
                bench_runner.clear()
                sequences = None
                prefill_latencies = []
                decode_latencies = []
                checks = []
                initial_kv_available = runner.token_to_kv_pool_allocator.available_size()
                initial_request_slots_available = runner.req_to_token_pool.available_size()
                previous_kv_available = initial_kv_available
                previous_request_slots_available = initial_request_slots_available
                for step_index in range(arguments.output_length):
                    if step_index == 0:
                        input_ids, requests = make_requests(
                            batch_size,
                            context_length,
                            config.vocab_size,
                            arguments.output_length,
                            case_seed + repetition,
                        )
                        sequences = input_ids.tolist()
                        forward = timed_forward(extend, requests, runner)
                        next_token_ids, next_token_logits, batch = forward["result"]
                        phase = "prefill"
                        expected_kv_available = initial_kv_available - batch_size * context_length
                        expected_request_slots_available = initial_request_slots_available - batch_size
                    else:
                        next_tokens = torch.tensor(
                            [sequence[-1] for sequence in sequences],
                            dtype=torch.int64,
                            device=device,
                        )
                        forward = timed_forward(decode, next_tokens, batch, runner)
                        next_token_ids, next_token_logits = forward["result"]
                        phase = "decode"
                        expected_kv_available = previous_kv_available - batch_size
                        expected_request_slots_available = previous_request_slots_available
                    kv_available_before = previous_kv_available
                    request_slots_available_before = previous_request_slots_available
                    kv_available_after = runner.token_to_kv_pool_allocator.available_size()
                    request_slots_available_after = runner.req_to_token_pool.available_size()
                    state_check_passed = (
                        kv_available_after == expected_kv_available
                        and request_slots_available_after == expected_request_slots_available
                    )
                    if not state_check_passed:
                        raise RuntimeError(
                            "Allocator state mismatch: "
                            f"phase={phase}, step={step_index}, batch={batch_size}, "
                            f"context={context_length}, repetition={repetition}, "
                            f"kv_before={kv_available_before}, kv_after={kv_available_after}, "
                            f"expected_kv={expected_kv_available}, "
                            f"requests_before={request_slots_available_before}, "
                            f"requests_after={request_slots_available_after}, "
                            f"expected_requests={expected_request_slots_available}"
                        )
                    previous_kv_available = kv_available_after
                    previous_request_slots_available = request_slots_available_after
                    control_output = control_logits(control, sequences, device)
                    control_tokens = control_output.argmax(dim=-1)
                    sglang_tokens = next_token_ids.to(torch.int64).tolist()
                    control_token_list = control_tokens.tolist()
                    tokens_equal = sglang_tokens == control_token_list
                    logits_difference = (next_token_logits.float() - control_output.float()).abs()
                    max_absolute_difference = float(logits_difference.max().item())
                    tolerance = arguments.logit_atol + arguments.logit_rtol * float(control_output.abs().max().item())
                    logits_within_tolerance = max_absolute_difference <= tolerance
                    control_top_two = torch.topk(control_output, k=2, dim=-1).values
                    control_top_margin = (control_top_two[:, 0] - control_top_two[:, 1]).abs()
                    near_tie = bool((control_top_margin <= tolerance).all().item())
                    output_check_passed = tokens_equal or near_tie
                    if not logits_within_tolerance or not output_check_passed:
                        raise RuntimeError(
                            "Independent Torch control mismatch: "
                            f"phase={phase}, step={step_index}, batch={batch_size}, "
                            f"context={context_length}, repetition={repetition}, "
                            f"sglang={sglang_tokens}, control={control_token_list}, "
                            f"max_abs_diff={max_absolute_difference:.6f}, tolerance={tolerance:.6f}, "
                            f"control_top_margin={float(control_top_margin.max().item()):.6f}"
                        )
                    phase_predicted_activation_bytes = predicted_activation_bytes(
                        config, batch_size, context_length + step_index, phase
                    )
                    for sequence, token in zip(sequences, sglang_tokens):
                        sequence.append(token)
                    if measured:
                        if phase == "prefill":
                            prefill_latencies.append(forward["latency_seconds"])
                        else:
                            decode_latencies.append(forward["latency_seconds"])
                    checks.append(
                        {
                            "phase": phase,
                            "step": step_index,
                            "tokens_equal": tokens_equal,
                            "near_tie": near_tie,
                            "sglang_tokens": sglang_tokens,
                            "control_tokens": control_token_list,
                            "control_top_margin": float(control_top_margin.max().item()),
                            "state_check_passed": state_check_passed,
                            "kv_available_before": kv_available_before,
                            "kv_available_after": kv_available_after,
                            "expected_kv_available_after": expected_kv_available,
                            "request_slots_available_before": request_slots_available_before,
                            "request_slots_available_after": request_slots_available_after,
                            "expected_request_slots_available_after": expected_request_slots_available,
                            "prediction_error_bytes": (
                                phase_predicted_activation_bytes
                                - forward["peak_allocated_delta_bytes"]
                            ),
                            "prediction_overhead_ratio": (
                                phase_predicted_activation_bytes
                                / max(1, forward["peak_allocated_delta_bytes"])
                            ),
                            "max_abs_logit_difference": max_absolute_difference,
                            "tolerance": tolerance,
                            "predicted_activation_bytes": phase_predicted_activation_bytes,
                            "measured_live_allocated_bytes": forward["post_live_allocated_bytes"],
                            "measured_peak_allocated_bytes": forward["peak_allocated_bytes"],
                            "measured_peak_allocated_delta_bytes": forward["peak_allocated_delta_bytes"],
                            "measured_live_reserved_bytes": forward["post_live_reserved_bytes"],
                            "latency_seconds": forward["latency_seconds"],
                        }
                    )
                if measured:
                    case["repetitions"].append(
                        {
                            "repetition": repetition - arguments.warmups,
                            "prefill_latency_seconds": prefill_latencies[0],
                            "decode_latency_seconds": decode_latencies,
                            "median_decode_latency_seconds": float(np.median(decode_latencies)),
                            "p90_decode_latency_seconds": float(np.percentile(decode_latencies, 90)),
                            "checks": checks,
                        }
                    )
                bench_runner.clear()
            case["predicted_prefill_activation_bytes"] = predicted_activation_bytes(
                config, batch_size, context_length, "prefill"
            )
            case["predicted_decode_activation_bytes"] = predicted_activation_bytes(
                config, batch_size, context_length, "decode"
            )
            case["predicted_kv_workspace_bytes"] = predicted_kv_bytes
            case["predicted_request_workspace_bytes"] = request_pool_bytes
            case["predicted_total_workspace_bytes"] = (
                predicted_kv_bytes + request_pool_bytes
            )
            results.append(case)

    all_decode_latencies = [
        latency
        for case in results
        for repetition in case["repetitions"]
        for latency in repetition["decode_latency_seconds"]
    ]
    report = {
        "schema": "sglang-synthetic-memory-scaling-v1",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(Path(__file__).resolve().parents[3]), text=True
        ).strip(),
        "config_path": str(config_dir),
        "config": config.to_dict(),
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "device_count": torch.cuda.device_count(),
            "capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
        },
        "python": sys.executable,
        "torch": {
            "version": torch.__version__,
            "path": torch.__file__,
            "hip": torch.version.hip,
        },
        "sglang": {
            "version": __import__("sglang").__version__,
            "path": __import__("sglang").__file__,
            "sgl_kernel_path": module_path("sgl_kernel"),
        },
        "server_args": {
            "model_path": str(config_dir),
            "load_format": "dummy",
            "dtype": "bfloat16",
            "attention_backend": "triton",
            "disable_cuda_graph": True,
            "mem_fraction_static": 0.02,
            "max_total_tokens": 4096,
            "max_running_requests": 16,
            "skip_tokenizer_init": True,
        },
        "weight_bytes": {
            "sglang": weights_bytes,
            "independent_torch_control": control_weights_bytes,
        },
        "kv_workspace_bytes": {
            "key": kv_size[0],
            "value": kv_size[1],
            "total": predicted_kv_bytes,
        },
        "request_workspace_bytes": request_pool_bytes,
        "max_total_num_tokens": runner.max_total_num_tokens,
        "workload_cases": results,
        "summary": {
            "case_count": len(results),
            "median_decode_latency_seconds": float(np.median(all_decode_latencies)),
            "p90_decode_latency_seconds": float(np.percentile(all_decode_latencies, 90)),
            "max_measured_live_allocated_bytes": max(
                check["measured_live_allocated_bytes"]
                for case in results
                for repetition in case["repetitions"]
                for check in repetition["checks"]
            ),
            "max_measured_peak_allocated_bytes": max(
                check["measured_peak_allocated_bytes"]
                for case in results
                for repetition in case["repetitions"]
                for check in repetition["checks"]
            ),
        },
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("w") as output:
        json.dump(report, output, indent=2, default=json_default)
    print(json.dumps(report["summary"], indent=2))
    print(f"results={arguments.output}")

    destroy_model_parallel()
    destroy_distributed_environment()


if __name__ == "__main__":
    main()
