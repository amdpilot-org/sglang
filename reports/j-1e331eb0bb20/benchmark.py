#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from sglang.benchmark.one_batch import (
    _TorchBenchRunner,
    decode,
    extend,
    prepare_synthetic_inputs_for_latency_test,
)
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.distributed.parallel_state_wrapper import ParallelState
from sglang.srt.model_executor.forward_context import ForwardContext, set_forward_context
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.runtime_context import publish
from sglang.srt.server_args import PortArgs, ServerArgs


MODEL_DIR = Path("/tmp/sglang-cache-j-1e331eb0bb20/minimal-llama")
RESULT_DIR = Path("/job/sglang/reports/j-1e331eb0bb20/results")
CONFIG = {
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "hidden_size": 512,
    "intermediate_size": 1024,
    "num_hidden_layers": 4,
    "num_attention_heads": 8,
    "num_key_value_heads": 4,
    "vocab_size": 512,
    "max_position_embeddings": 512,
    "rms_norm_eps": 1e-5,
    "rope_theta": 10000.0,
    "torch_dtype": "float16",
    "tie_word_embeddings": False,
}
WARMUP_ITERATIONS = 5
MEASURED_ITERATIONS = 20
RTOL = 5e-2
ATOL = 5e-2


def write_config():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (MODEL_DIR / "config.json").write_text(json.dumps(CONFIG, indent=2))


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_server_args(backend, load_format):
    return ServerArgs(
        model_path=str(MODEL_DIR),
        tokenizer_path=str(MODEL_DIR),
        load_format=load_format,
        attention_backend=backend,
        disable_cuda_graph=True,
        mem_fraction_static=0.05,
        context_length=512,
        dtype="float16",
        host="127.0.0.1",
        port=0,
        tp_size=1,
    )


def build_runner(backend, load_format):
    server_args = build_server_args(backend, load_format)
    port_args = PortArgs.init_new(server_args)
    publish(server_args, role="scheduler")
    model_config = ModelConfig.from_server_args(server_args)
    runner = ModelRunner(
        model_config=model_config,
        mem_fraction_static=server_args.mem_fraction_static,
        gpu_id=0,
        ps=ParallelState.trivial(tp_size=1),
        nccl_port=port_args.nccl_port,
        server_args=server_args,
    )
    runner.alloc_memory_pool()
    runner.init_attention_backends()
    runner.init_cuda_graphs()
    set_forward_context(ForwardContext(attn_backend=runner.attn_backend))
    return _TorchBenchRunner(runner)


def generate_weights():
    write_config()
    torch.manual_seed(394)
    runner = build_runner("triton", "dummy")
    state_dict = {
        name: value.detach().cpu().contiguous()
        for name, value in runner.torch_runner.model.state_dict().items()
    }
    weight_path = MODEL_DIR / "model.safetensors"
    save_file(state_dict, str(weight_path), metadata={"format": "pt"})
    weight_bytes = weight_path.stat().st_size
    assert weight_bytes < 4 * 1024**3
    return {
        "weight_path": str(weight_path),
        "weight_bytes": weight_bytes,
        "weight_sha256": sha256(weight_path),
        "parameter_count": sum(value.numel() for value in state_dict.values()),
    }


def rms_norm(hidden, weight, eps):
    original_dtype = hidden.dtype
    hidden = hidden.float()
    variance = hidden.pow(2).mean(dim=-1, keepdim=True)
    hidden = hidden * torch.rsqrt(variance + eps)
    hidden = (hidden * weight.float()).to(original_dtype)
    return hidden


def apply_rope(tensor, cos, sin):
    cos = cos.to(tensor.dtype)
    sin = sin.to(tensor.dtype)
    first, second = tensor.chunk(2, dim=-1)
    return torch.cat((first * cos - second * sin, second * cos + first * sin), dim=-1)


def independent_reference(input_ids):
    weights = load_file(str(MODEL_DIR / "model.safetensors"), device="cuda:0")
    sequence_length = input_ids.shape[1]
    hidden = F.embedding(input_ids, weights["model.embed_tokens.weight"])
    inverse_frequency = 1.0 / (
        10000.0
        ** (
            torch.arange(0, 64, 2, dtype=torch.float32, device="cuda:0")
            / 64
        )
    )
    positions = torch.arange(sequence_length, dtype=torch.float32, device="cuda:0")
    frequencies = torch.outer(positions, inverse_frequency)
    cos = frequencies.cos()[:, None, :]
    sin = frequencies.sin()[:, None, :]
    for layer_index in range(CONFIG["num_hidden_layers"]):
        prefix = f"model.layers.{layer_index}"
        residual = hidden
        hidden = rms_norm(
            hidden,
            weights[f"{prefix}.input_layernorm.weight"],
            CONFIG["rms_norm_eps"],
        )
        qkv = hidden @ weights[f"{prefix}.self_attn.qkv_proj.weight"].T
        query, key, value = qkv.split([512, 256, 256], dim=-1)
        query = query.view(sequence_length, 8, 64)
        key = key.view(sequence_length, 4, 64)
        value = value.view(sequence_length, 4, 64)
        query = apply_rope(query, cos, sin)
        key = apply_rope(key, cos, sin)
        key = key.repeat_interleave(2, dim=1)
        value = value.repeat_interleave(2, dim=1)
        attention = F.scaled_dot_product_attention(
            query.transpose(0, 1),
            key.transpose(0, 1),
            value.transpose(0, 1),
            is_causal=True,
            scale=64**-0.5,
        )
        attention = attention.transpose(0, 1).reshape(sequence_length, 512)
        hidden = residual + attention @ weights[f"{prefix}.self_attn.o_proj.weight"].T
        residual = hidden
        hidden = rms_norm(
            hidden,
            weights[f"{prefix}.post_attention_layernorm.weight"],
            CONFIG["rms_norm_eps"],
        )
        gate_up = hidden @ weights[f"{prefix}.mlp.gate_up_proj.weight"].T
        gate, up = gate_up.chunk(2, dim=-1)
        hidden = residual + (F.silu(gate) * up) @ weights[f"{prefix}.mlp.down_proj.weight"].T
    hidden = rms_norm(hidden, weights["model.norm.weight"], CONFIG["rms_norm_eps"])
    logits = hidden @ weights["lm_head.weight"].T
    return logits[:, -1, :]


def make_inputs(batch_size, sequence_length, seed):
    generator = np.random.default_rng(seed)
    return generator.integers(0, CONFIG["vocab_size"], (batch_size, sequence_length), dtype=np.int64)


def profile_kernel_names(runner, input_ids):
    reqs = prepare_synthetic_inputs_for_latency_test(
        input_ids.shape[0], input_ids.shape[1], input_ids.tolist()
    )
    runner.clear()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as profiler:
        runner.extend(reqs)
        torch.cuda.synchronize()
    events = profiler.key_averages()
    kernels = []
    for event in events:
        if event.self_device_time_total > 0:
            kernels.append(
                {
                    "name": event.key,
                    "self_device_time_us": event.self_device_time_total,
                }
            )
    kernels.sort(key=lambda item: item["self_device_time_us"], reverse=True)
    return kernels[:20]


def compare_to_reference(actual, reference):
    actual = actual.float()
    reference = reference.float()
    difference = (actual - reference).abs()
    max_abs = difference.max().item()
    max_ref = reference.abs().max().item()
    relative = max_abs / max_ref
    try:
        torch.testing.assert_close(actual, reference, rtol=RTOL, atol=ATOL)
        passed = True
    except AssertionError:
        passed = False
    return {
        "gate": f"torch.testing.assert_close(rtol={RTOL}, atol={ATOL})",
        "passed": passed,
        "max_abs_difference": max_abs,
        "max_reference_abs": max_ref,
        "relative_max_difference": relative,
    }


def timing_summary(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": float(array.mean() * 1000),
        "median_ms": float(np.median(array) * 1000),
        "min_ms": float(array.min() * 1000),
        "max_ms": float(array.max() * 1000),
        "iterations": len(array),
    }


def run_backend(backend):
    write_config()
    runner = build_runner(backend, "safetensors")
    correctness_inputs = make_inputs(1, 32, 394)
    reference = independent_reference(torch.tensor(correctness_inputs, device="cuda:0"))
    reqs = __import__(
        "sglang.benchmark.one_batch", fromlist=["prepare_synthetic_inputs_for_latency_test"]
    ).prepare_synthetic_inputs_for_latency_test(
        1, 32, correctness_inputs.tolist()
    )
    runner.clear()
    _, actual, _ = runner.extend(reqs)
    correctness = compare_to_reference(actual, reference)
    kernels = profile_kernel_names(runner, correctness_inputs)

    cases = []
    for case_name, batch_size, sequence_length, mode in [
        ("prefill_1x128", 1, 128, "prefill"),
        ("prefill_4x64", 4, 64, "prefill"),
        ("decode_4x64", 4, 64, "decode"),
    ]:
        latencies = []
        for iteration in range(WARMUP_ITERATIONS + MEASURED_ITERATIONS):
            seed = 1000 + iteration
            inputs = make_inputs(batch_size, sequence_length, seed)
            reqs = __import__(
                "sglang.benchmark.one_batch",
                fromlist=["prepare_synthetic_inputs_for_latency_test"],
            ).prepare_synthetic_inputs_for_latency_test(
                batch_size, sequence_length, inputs.tolist()
            )
            runner.clear()
            torch.cuda.synchronize()
            start = time.perf_counter()
            if mode == "prefill":
                _, _, batch = runner.extend(reqs)
            else:
                _, _, batch = runner.extend(reqs)
                decode_tokens = torch.tensor(
                    make_inputs(batch_size, 1, seed + 10000)[:, 0],
                    device="cuda:0",
                    dtype=torch.int64,
                )
                runner.decode(decode_tokens, batch)
            torch.cuda.synchronize()
            latency = time.perf_counter() - start
            if iteration >= WARMUP_ITERATIONS:
                latencies.append(latency)
            runner.cleanup(batch)
        cases.append(
            {
                "case": case_name,
                "batch_size": batch_size,
                "sequence_length": sequence_length,
                "mode": mode,
                "warmup_iterations": WARMUP_ITERATIONS,
                "measured_iterations": MEASURED_ITERATIONS,
                "timing": timing_summary(latencies),
            }
        )

    memory_allocated = torch.cuda.memory_allocated()
    max_memory_allocated = torch.cuda.max_memory_allocated()
    assert memory_allocated < 48 * 1024**3
    result = {
        "backend": backend,
        "attention_backend_class": type(runner.torch_runner.attn_backend).__name__,
        "attention_backend_source": __import__("inspect").getsourcefile(
            type(runner.torch_runner.attn_backend)
        ),
        "sglang_path": __import__("sglang").__file__,
        "triton_path": __import__("triton").__file__,
        "aiter_path": __import__("aiter").__file__,
        "torch_path": torch.__file__,
        "gpu_name": torch.cuda.get_device_name(0),
        "gcn_arch_name": torch.cuda.get_device_properties(0).gcnArchName,
        "torch_version": torch.__version__,
        "hip_version": torch.version.hip,
        "correctness": correctness,
        "kernel_names": kernels,
        "cases": cases,
        "memory_allocated_bytes": memory_allocated,
        "max_memory_allocated_bytes": max_memory_allocated,
    }
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULT_DIR / f"{backend}.json"
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    torch.save(actual.cpu(), RESULT_DIR / f"{backend}_correctness_logits.pt")
    return result


def compare_backends():
    triton = json.loads((RESULT_DIR / "triton.json").read_text())
    aiter = json.loads((RESULT_DIR / "aiter.json").read_text())
    triton_logits = torch.load(RESULT_DIR / "triton_correctness_logits.pt")
    aiter_logits = torch.load(RESULT_DIR / "aiter_correctness_logits.pt")
    difference = (triton_logits.float() - aiter_logits.float()).abs()
    result = {
        "backends": ["triton", "aiter"],
        "max_abs_difference": difference.max().item(),
        "max_triton_abs": triton_logits.float().abs().max().item(),
        "relative_max_difference": difference.max().item()
        / triton_logits.float().abs().max().item(),
        "triton": triton,
        "aiter": aiter,
    }
    (RESULT_DIR / "comparison.json").write_text(
        json.dumps(result, indent=2, sort_keys=True)
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["generate", "triton", "aiter", "compare"])
    args = parser.parse_args()
    if args.action == "generate":
        print(json.dumps(generate_weights(), indent=2, sort_keys=True))
    elif args.action == "compare":
        print(json.dumps(compare_backends(), indent=2, sort_keys=True))
    else:
        print(json.dumps(run_backend(args.action), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
