#!/usr/bin/env python3
"""Bounded MI300X Llama reduced-ModelRunner benchmark.

The benchmark generates a tiny local Llama checkpoint with random weights, runs
real SGLang offline Engine forwards, and compares each case with an independent
plain-Torch implementation. No tokenizer or network checkpoint is used.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file
from transformers import LlamaConfig, LlamaForCausalLM


SEED = 35003
VOCAB_SIZE = 512
HIDDEN_SIZE = 256
INTERMEDIATE_SIZE = 512
NUM_HIDDEN_LAYERS = 2
NUM_ATTENTION_HEADS = 4
NUM_KEY_VALUE_HEADS = 2
HEAD_DIM = 64
MAX_POSITION_EMBEDDINGS = 512
ROPE_THETA = 10000.0
RMS_NORM_EPS = 1e-5
DTYPE = "bfloat16"
CASE_LENGTHS = (8, 32, 64, 128, 256, 480)
CONFIGURATIONS = ("triton", "torch_native", "aiter", "wave")
WARMUP_FORWARDS = 2
MEASURED_FORWARDS = 10
TOP_K = 5
LOGPROB_ABS_GATE = 0.05
WEIGHT_LIMIT_BYTES = 4 * 1024**3
LIVE_ALLOCATION_LIMIT_BYTES = 48 * 1024**3
WALL_LIMIT_SECONDS = 7200
MODEL_DIR = Path("/tmp/sglang-cache-j-9a07dd1052cc/model")
IMAGE = {
    "qualified_image": "amdpilotv2/open-job-mi300:jit-config-readable-35122-260909",
    "local_image_id_sha256": "dfc9419089c338b5712da4841768b38b1ab79f3da41f8c58c3cd4dfcc1147ff1",
}


def deterministic_ids(length: int) -> list[int]:
    return [(SEED + 101 * index + 17 * index * index) % VOCAB_SIZE for index in range(length)]


def generate_model() -> dict[str, Any]:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    config = LlamaConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        num_hidden_layers=NUM_HIDDEN_LAYERS,
        num_attention_heads=NUM_ATTENTION_HEADS,
        num_key_value_heads=NUM_KEY_VALUE_HEADS,
        max_position_embeddings=MAX_POSITION_EMBEDDINGS,
        rms_norm_eps=RMS_NORM_EPS,
        tie_word_embeddings=False,
        torch_dtype=torch.bfloat16,
    )
    torch.manual_seed(SEED)
    model = LlamaForCausalLM(config).to(torch.bfloat16)
    model.save_pretrained(MODEL_DIR, safe_serialization=True)
    parameter_bytes = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
    del model
    torch.cuda.empty_cache()
    safetensors_file_bytes = (MODEL_DIR / "model.safetensors").stat().st_size
    return {
        "model_dir": str(MODEL_DIR),
        "parameter_bytes": parameter_bytes,
        "safetensors_file_bytes": safetensors_file_bytes,
        "weight_bytes": safetensors_file_bytes,
        "weight_limit_bytes": WEIGHT_LIMIT_BYTES,
        "weight_limit_pass": safetensors_file_bytes < WEIGHT_LIMIT_BYTES,
    }


def _rms_norm(hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return hidden * torch.rsqrt(hidden.pow(2).mean(-1, keepdim=True) + RMS_NORM_EPS) * weight.float()


def independent_torch_control(input_ids: list[int], weights: dict[str, torch.Tensor]) -> list[dict[str, float]]:
    """Independent plain-Torch Llama forward used only as a numerical control."""
    with torch.no_grad():
        tokens = torch.tensor(input_ids, dtype=torch.long)
        hidden = weights["model.embed_tokens.weight"][tokens].float().unsqueeze(0)
        sequence_length = tokens.shape[0]
        inverse_frequency = 1.0 / (
            ROPE_THETA
            ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM)
        )
        positions = torch.arange(sequence_length, dtype=torch.float32)
        frequency = torch.outer(positions, inverse_frequency)
        cosine = frequency.cos().unsqueeze(0).unsqueeze(2)
        sine = frequency.sin().unsqueeze(0).unsqueeze(2)
        causal_mask = torch.full((sequence_length, sequence_length), float("-inf")).triu(1)

        for layer_index in range(NUM_HIDDEN_LAYERS):
            prefix = f"model.layers.{layer_index}."
            normalized = _rms_norm(hidden, weights[prefix + "input_layernorm.weight"])
            query = torch.einsum(
                "bsh,oh->bso", normalized, weights[prefix + "self_attn.q_proj.weight"].float()
            ).view(1, sequence_length, NUM_ATTENTION_HEADS, HEAD_DIM)
            key = torch.einsum(
                "bsh,oh->bso", normalized, weights[prefix + "self_attn.k_proj.weight"].float()
            ).view(1, sequence_length, NUM_KEY_VALUE_HEADS, HEAD_DIM)
            value = torch.einsum(
                "bsh,oh->bso", normalized, weights[prefix + "self_attn.v_proj.weight"].float()
            ).view(1, sequence_length, NUM_KEY_VALUE_HEADS, HEAD_DIM)

            def rotary(tensor: torch.Tensor) -> torch.Tensor:
                first, second = tensor[..., : HEAD_DIM // 2], tensor[..., HEAD_DIM // 2 :]
                return torch.cat([first * cosine - second * sine, first * sine + second * cosine], dim=-1)

            query, key = rotary(query), rotary(key)
            attention_outputs = []
            for head_index in range(NUM_ATTENTION_HEADS):
                kv_head = head_index // (NUM_ATTENTION_HEADS // NUM_KEY_VALUE_HEADS)
                scores = (
                    query[0, :, head_index, :] @ key[0, :, kv_head, :].T
                ) / math.sqrt(HEAD_DIM)
                attention = torch.softmax(scores + causal_mask, dim=-1)
                attention_outputs.append(attention @ value[0, :, kv_head, :])
            attention_output = torch.stack(attention_outputs, dim=1).reshape(
                1, sequence_length, NUM_ATTENTION_HEADS * HEAD_DIM
            )
            projected_attention = torch.einsum(
                "bsh,oh->bso", attention_output, weights[prefix + "self_attn.o_proj.weight"].float()
            )
            hidden = hidden + projected_attention

            normalized = _rms_norm(hidden, weights[prefix + "post_attention_layernorm.weight"])
            gate = torch.einsum(
                "bsh,oh->bso", normalized, weights[prefix + "mlp.gate_proj.weight"].float()
            )
            up = torch.einsum(
                "bsh,oh->bso", normalized, weights[prefix + "mlp.up_proj.weight"].float()
            )
            mlp = torch.einsum(
                "bsh,oh->bso",
                torch.nn.functional.silu(gate) * up,
                weights[prefix + "mlp.down_proj.weight"].float(),
            )
            hidden = hidden + mlp

        hidden = _rms_norm(hidden, weights["model.norm.weight"])
        logits = torch.einsum("bsh,vh->bsv", hidden, weights["lm_head.weight"].float())
        logprobabilities = torch.log_softmax(logits[0, -1], dim=-1)
        top = torch.topk(logprobabilities, TOP_K)
        return [
            {"token": int(token), "logprob": float(logprob)}
            for token, logprob in zip(top.indices.tolist(), top.values.tolist())
        ]


def _summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p95": ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)],
        "min": min(values),
        "max": max(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def _device_used_bytes() -> int:
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return total_bytes - free_bytes


def _parse_top_logprobs(output: dict[str, Any]) -> list[dict[str, float]]:
    entries = output["meta_info"]["output_top_logprobs"][0]
    return [
        {"token": int(entry[1]), "logprob": float(entry[0])}
        for entry in entries
    ]


def run_configuration(configuration: str) -> dict[str, Any]:
    from sglang.srt.entrypoints.engine import Engine

    weights = load_file(MODEL_DIR / "model.safetensors")
    engine = None
    started = time.perf_counter()
    results = []
    max_device_used_bytes = 0
    numerical_pass = True
    try:
        engine = Engine(
            model_path=str(MODEL_DIR),
            skip_tokenizer_init=True,
            attention_backend=configuration,
            dtype=DTYPE,
            mem_fraction_static=0.20,
            context_length=MAX_POSITION_EMBEDDINGS,
            disable_cuda_graph=True,
            disable_radix_cache=True,
            log_level="error",
        )
        for sequence_length in CASE_LENGTHS:
            input_ids = deterministic_ids(sequence_length)
            control = independent_torch_control(input_ids, weights)
            for _ in range(WARMUP_FORWARDS):
                engine.generate(
                    input_ids=input_ids,
                    sampling_params={"max_new_tokens": 1, "temperature": 0.0, "ignore_eos": True},
                    return_logprob=True,
                    logprob_start_len=sequence_length - 1,
                    top_logprobs_num=TOP_K,
                )
            latencies = []
            observed = []
            for _ in range(MEASURED_FORWARDS):
                output = engine.generate(
                    input_ids=input_ids,
                    sampling_params={"max_new_tokens": 1, "temperature": 0.0, "ignore_eos": True},
                    return_logprob=True,
                    logprob_start_len=sequence_length - 1,
                    top_logprobs_num=TOP_K,
                )
                latencies.append(float(output["meta_info"]["e2e_latency"]))
                observed.append(_parse_top_logprobs(output))
                max_device_used_bytes = max(max_device_used_bytes, _device_used_bytes())

            control_by_token = {entry["token"]: entry["logprob"] for entry in control}
            top1_pass = all(
                observed_run[0]["token"] == control[0]["token"] for observed_run in observed
            )
            top_k_set_pass = all(
                {entry["token"] for entry in observed_run} == set(control_by_token)
                for observed_run in observed
            )
            token_pass = top1_pass and top_k_set_pass
            max_logprob_error = max(
                abs(entry["logprob"] - control_by_token[entry["token"]])
                for observed_run in observed
                for entry in observed_run
                if entry["token"] in control_by_token
            )
            case_pass = token_pass and max_logprob_error <= LOGPROB_ABS_GATE
            numerical_pass = numerical_pass and case_pass
            results.append(
                {
                    "sequence_length": sequence_length,
                    "batch_size": 1,
                    "input_ids": input_ids,
                    "control_top_k": control,
                    "observed_top_k": observed,
                    "top1_pass": top1_pass,
                    "top_k_set_pass": top_k_set_pass,
                    "token_pass": token_pass,
                    "max_logprob_abs_error": max_logprob_error,
                    "numerical_pass": case_pass,
                    "latencies_ms": [value * 1000.0 for value in latencies],
                    "latency_summary_ms": _summary([value * 1000.0 for value in latencies]),
                }
            )
    except Exception as error:
        return {
            "configuration": configuration,
            "supported": False,
            "error": f"{type(error).__name__}: {error}",
            "elapsed_seconds": time.perf_counter() - started,
        }
    finally:
        if engine is not None:
            engine.shutdown()

    passing_medians = [case["latency_summary_ms"]["median"] for case in results] if numerical_pass else []
    return {
        "configuration": configuration,
        "supported": True,
        "numerical_pass": numerical_pass,
        "elapsed_seconds": time.perf_counter() - started,
        "max_device_used_bytes": max_device_used_bytes,
        "live_allocation_limit_bytes": LIVE_ALLOCATION_LIMIT_BYTES,
        "live_allocation_pass": max_device_used_bytes < LIVE_ALLOCATION_LIMIT_BYTES,
        "case_median_ms": statistics.median(passing_medians) if passing_medians else None,
        "cases": results,
    }


def _source_metadata() -> dict[str, Any]:
    import sglang
    import sgl_kernel

    native_candidates = list(Path(sgl_kernel.__file__).parent.glob("*.so"))
    return {
        "sglang_import_path": sglang.__file__,
        "sgl_kernel_import_path": sgl_kernel.__file__,
        "sgl_kernel_native_path": str(native_candidates[0]) if native_candidates else None,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }


def _gpu_metadata() -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(0)
    return {
        "name": properties.name,
        "gcn_arch": properties.gcnArchName,
        "uuid": str(properties.uuid),
        "total_memory_bytes": properties.total_memory,
        "device_count": torch.cuda.device_count(),
    }


def run_all(output_path: Path) -> None:
    started = time.perf_counter()
    model_info = generate_model()
    configuration_results = []
    for configuration in CONFIGURATIONS:
        temporary = Path(f"/tmp/sglang-cache-j-9a07dd1052cc/{configuration}.json")
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--configuration",
            configuration,
            "--output",
            str(temporary),
        ]
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        if completed.returncode != 0:
            configuration_results.append(
                {
                    "configuration": configuration,
                    "supported": False,
                    "error": completed.stderr.strip()[-4000:],
                    "return_code": completed.returncode,
                }
            )
        else:
            with temporary.open() as handle:
                configuration_results.append(json.load(handle))

    passing = [result for result in configuration_results if result.get("supported") and result.get("numerical_pass")]
    baseline = next((result for result in passing if result["configuration"] == "triton"), None)
    for result in passing:
        ratios = []
        for case, baseline_case in zip(result["cases"], baseline["cases"] if baseline else []):
            ratios.append(case["latency_summary_ms"]["median"] / baseline_case["latency_summary_ms"]["median"])
        result["geometric_mean_latency_ratio_vs_triton"] = math.exp(statistics.fmean(math.log(value) for value in ratios)) if ratios else None

    result = {
        "label": "real SGLang offline Engine reduced-ModelRunner study",
        "campaign": "repo-e2e-20260909",
        "upstream_context": {
            "issue": 35003,
            "title": "AMD Development Roadmap (2026 Q3)",
            "url": "https://github.com/sgl-project/sglang/issues/35003",
            "comments": 0,
            "related_changes": [19975, 23388, 25090, 33939],
        },
        "image": IMAGE,
        "gpu": _gpu_metadata(),
        "source": _source_metadata(),
        "model": model_info,
        "dimensions": {
            "vocab_size": VOCAB_SIZE,
            "hidden_size": HIDDEN_SIZE,
            "intermediate_size": INTERMEDIATE_SIZE,
            "num_hidden_layers": NUM_HIDDEN_LAYERS,
            "num_attention_heads": NUM_ATTENTION_HEADS,
            "num_key_value_heads": NUM_KEY_VALUE_HEADS,
            "head_dim": HEAD_DIM,
            "max_position_embeddings": MAX_POSITION_EMBEDDINGS,
            "dtype": DTYPE,
            "case_lengths": list(CASE_LENGTHS),
            "batch_size": 1,
        },
        "configurations": list(CONFIGURATIONS),
        "forward_counts": {
            "warmup_per_case": WARMUP_FORWARDS,
            "measured_per_case": MEASURED_FORWARDS,
            "total_real_forwards": len(CONFIGURATIONS) * len(CASE_LENGTHS) * (WARMUP_FORWARDS + MEASURED_FORWARDS),
        },
        "numerical_gate": {
            "top_k_tokens_exact": True,
            "max_logprob_abs_error": LOGPROB_ABS_GATE,
        },
        "timing_method": "SGLang meta_info.e2e_latency for one-token offline Engine generate; 2 warmups and 10 measured calls per case",
        "limits": {
            "wall_seconds": WALL_LIMIT_SECONDS,
            "weight_bytes": WEIGHT_LIMIT_BYTES,
            "live_allocation_bytes": LIVE_ALLOCATION_LIMIT_BYTES,
            "max_workload_cases": 6,
            "max_configurations": 4,
        },
        "results": configuration_results,
        "all_numerical_pass": bool(passing) and all(result["numerical_pass"] for result in passing),
        "wall_elapsed_seconds": time.perf_counter() - started,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configuration", choices=[*CONFIGURATIONS, "all"], default="all")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.configuration == "all":
        run_all(args.output)
    else:
        if not (MODEL_DIR / "model.safetensors").exists():
            generate_model()
        result = run_configuration(args.configuration)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as handle:
            json.dump(result, handle, indent=2)
            handle.write("\n")


if __name__ == "__main__":
    main()
