#!/usr/bin/env python3

import argparse
import json
import os
import subprocess
import sys
import time
from array import array
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM

from sglang.benchmark.one_batch import TreeCacheNamespace
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.distributed.parallel_state_wrapper import ParallelState
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardBatch, ForwardMode
from sglang.srt.model_executor.forward_context import ForwardContext, set_forward_context
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.runtime_context import publish
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm


SEED = 30734
BATCH_SIZE = 2
INPUT_LEN = 512
NUM_LAYERS = 8
HIDDEN_SIZE = 512
INTERMEDIATE_SIZE = 1024
NUM_ATTENTION_HEADS = 8
NUM_KEY_VALUE_HEADS = 2
VOCAB_SIZE = 1024
MAX_POSITION_EMBEDDINGS = 4096
WARM_ITERS = 5
INTERNAL_ATOL = 1e-3
INTERNAL_RTOL = 1e-3
CONTROL_ATOL = 2e-2
CONTROL_RTOL = 2e-2


def git_output(command):
    return subprocess.check_output(command, cwd=Path(__file__).resolve().parents[2], text=True).strip()


def generate_model(model_dir):
    config = LlamaConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=NUM_ATTENTION_HEADS,
        num_key_value_heads=NUM_KEY_VALUE_HEADS,
        max_position_embeddings=MAX_POSITION_EMBEDDINGS,
        rms_norm_eps=1e-5,
        rope_theta=10000,
        tie_word_embeddings=False,
        dtype=torch.bfloat16,
    )
    torch.manual_seed(SEED)
    model = LlamaForCausalLM(config).to(torch.bfloat16)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.mul_(0.02).add_(0.001)
    model.save_pretrained(model_dir, safe_serialization=True)
    return config, sum(parameter.numel() for parameter in model.parameters())


def tensor_metrics(actual, expected, atol, rtol):
    actual = actual.float()
    expected = expected.float()
    difference = (actual - expected).abs()
    return {
        "shape": list(actual.shape),
        "dtype": str(actual.dtype),
        "max_abs_error": float(difference.max().item()),
        "mean_abs_error": float(difference.mean().item()),
        "allclose": bool(torch.allclose(actual, expected, atol=atol, rtol=rtol)),
        "all_finite": bool(torch.isfinite(actual).all().item()),
        "bitwise_equal": bool(torch.equal(actual, expected)),
        "different_elements": int((actual != expected).sum().item()),
    }


def make_batch(runner, model_config, input_ids, split):
    requests = []
    for row in input_ids.tolist():
        request = Req(
            rid=len(requests),
            origin_input_text="",
            origin_input_ids=array("q", row),
            sampling_params=SamplingParams(temperature=0.0, max_new_tokens=1),
        )
        request.full_untruncated_fill_ids = request.origin_input_ids
        request.logprob_start_len = -1
        request.set_extend_range(
            len(request.prefix_indices), len(request.full_untruncated_fill_ids)
        )
        requests.append(request)
    tree_cache = TreeCacheNamespace(
        page_size=1,
        device=runner.device,
        token_to_kv_pool_allocator=runner.token_to_kv_pool_allocator,
    )
    batch = ScheduleBatch.init_new(
        reqs=requests,
        req_to_token_pool=runner.req_to_token_pool,
        token_to_kv_pool_allocator=runner.token_to_kv_pool_allocator,
        tree_cache=tree_cache,
        model_config=model_config,
        enable_overlap=False,
        spec_algorithm=SpeculativeAlgorithm.NONE,
    )
    batch.prepare_for_extend()
    if batch.input_ids is None and batch.prefill_input_ids_cpu is not None:
        batch.input_ids = batch.prefill_input_ids_cpu.to(runner.device, non_blocking=True)
        batch.prefill_input_ids_cpu = None
    if split:
        batch.forward_mode = ForwardMode.SPLIT_PREFILL
    forward_batch = ForwardBatch.init_new(
        batch,
        runner,
        return_hidden_states_before_norm=False,
    )
    forward_batch.capture_hidden_mode = CaptureHiddenMode.FULL
    return forward_batch


def run_normal(runner, forward_batch):
    torch.cuda.synchronize()
    start = time.perf_counter()
    output = runner.forward(forward_batch).logits_output
    torch.cuda.synchronize()
    return output, time.perf_counter() - start


def run_split(runner, forward_batch, partition):
    forward_batch.split_index = 0
    forward_batch.hidden_states = None
    forward_batch.residual = None
    torch.cuda.synchronize()
    start = time.perf_counter()
    output = None
    for chunk_index, forward_count in enumerate(partition):
        result = runner.forward_split_prefill(
            forward_batch=forward_batch,
            reinit_attn_backend=(chunk_index == 0),
            forward_count=forward_count,
        )
        if result is not None:
            output = result
    torch.cuda.synchronize()
    return output, time.perf_counter() - start, forward_batch.split_index


def elapsed_statistics(values):
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "values": values,
        "mean": float(tensor.mean().item()),
        "median": float(tensor.median().item()),
        "min": float(tensor.min().item()),
        "max": float(tensor.max().item()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        default="/tmp/sglang-cache-j-c905c499f1be/llama-synthetic",
    )
    parser.add_argument("--warm-iters", type=int, default=WARM_ITERS)
    args = parser.parse_args()
    if not 1 <= args.warm_iters <= 10:
        raise ValueError("--warm-iters must be between 1 and 10")

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    if not (model_dir / "model.safetensors").exists():
        generated_config, parameter_count = generate_model(model_dir)
    else:
        generated_config = LlamaConfig.from_pretrained(model_dir, local_files_only=True)
        parameter_count = None

    server_args = ServerArgs(
        model_path=str(model_dir),
        tokenizer_path=str(model_dir),
        host="127.0.0.1",
        port=30000,
        tp_size=1,
        attention_backend="torch_native",
        disable_cuda_graph=True,
        disable_hybrid_swa_memory=True,
        mem_fraction_static=0.05,
        trust_remote_code=False,
    )
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

    generator = torch.Generator(device="cpu").manual_seed(SEED)
    input_ids = torch.randint(
        10,
        VOCAB_SIZE,
        (BATCH_SIZE, INPUT_LEN),
        dtype=torch.int32,
        generator=generator,
    )
    useful_tokens = BATCH_SIZE * INPUT_LEN
    useful_layer_tokens = useful_tokens * model_config.num_hidden_layers
    cases = [
        ("normal_unchunked", None),
        ("split_8", [8]),
        ("split_4_4", [4, 4]),
        ("split_2_2_2_2", [2, 2, 2, 2]),
        ("split_1x8", [1] * 8),
        ("split_5_3", [5, 3]),
    ]

    results = []
    normal_logits = None
    normal_hidden_states = None
    for case_index, (name, partition) in enumerate(cases):
        torch.cuda.reset_peak_memory_stats()
        if partition is None:
            cold_batch = make_batch(runner, model_config, input_ids, False)
            cold_output, cold_elapsed = run_normal(runner, cold_batch)
            final_split_index = None
        else:
            cold_batch = make_batch(runner, model_config, input_ids, True)
            cold_output, cold_elapsed, final_split_index = run_split(
                runner, cold_batch, partition
            )
        warm_elapsed = []
        warm_output = cold_output
        for _ in range(args.warm_iters):
            if partition is None:
                warm_batch = make_batch(runner, model_config, input_ids, False)
                warm_output, elapsed = run_normal(runner, warm_batch)
            else:
                warm_batch = make_batch(runner, model_config, input_ids, True)
                warm_output, elapsed, _ = run_split(runner, warm_batch, partition)
            warm_elapsed.append(elapsed)
        if case_index == 0:
            normal_logits = warm_output.next_token_logits.detach().clone()
            normal_hidden_states = warm_output.hidden_states.detach().clone()
        logits_metrics = tensor_metrics(
            warm_output.next_token_logits, normal_logits, INTERNAL_ATOL, INTERNAL_RTOL
        )
        hidden_metrics = tensor_metrics(
            warm_output.hidden_states, normal_hidden_states, INTERNAL_ATOL, INTERNAL_RTOL
        )
        elapsed_stats = elapsed_statistics(warm_elapsed)
        results.append(
            {
                "name": name,
                "partition": partition,
                "cold_elapsed_sec": cold_elapsed,
                "warm_elapsed": elapsed_stats,
                "useful_tokens": useful_tokens,
                "useful_layer_tokens": useful_layer_tokens,
                "useful_tokens_per_sec": useful_tokens / elapsed_stats["median"],
                "useful_layer_tokens_per_sec": useful_layer_tokens / elapsed_stats["median"],
                "final_split_index": final_split_index,
                "logits_vs_normal": logits_metrics,
                "final_hidden_states_vs_normal": hidden_metrics,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "allocated_bytes_after": torch.cuda.memory_allocated(),
                "reserved_bytes_after": torch.cuda.memory_reserved(),
            }
        )

    control = AutoModelForCausalLM.from_pretrained(
        model_dir,
        dtype=torch.bfloat16,
        device_map="cuda",
        local_files_only=True,
    ).eval()
    with torch.inference_mode():
        control_output = control(input_ids=input_ids.cuda()).logits[:, -1]
        control_hidden_states = (
            control.model(input_ids=input_ids.cuda(), use_cache=False)
            .last_hidden_state.reshape(-1, HIDDEN_SIZE)
        )
    control_logits_metrics = tensor_metrics(
        normal_logits, control_output, CONTROL_ATOL, CONTROL_RTOL
    )
    control_hidden_metrics = tensor_metrics(
        normal_hidden_states, control_hidden_states, CONTROL_ATOL, CONTROL_RTOL
    )

    import sglang
    import sglang.srt.model_executor.model_runner as model_runner_module
    import sgl_kernel
    import triton

    output = {
        "label": "real SGLang ModelRunner layer-chunk partition study",
        "base_commit": git_output(["git", "rev-parse", "HEAD"]),
        "branch": git_output(["git", "branch", "--show-current"]),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": {
            "path": str(model_dir),
            "architecture": generated_config.architectures[0],
            "parameter_count": sum(parameter.numel() for parameter in control.parameters()),
            "weight_file_bytes": (model_dir / "model.safetensors").stat().st_size,
            "dtype": str(model_config.dtype),
            "num_hidden_layers": model_config.num_hidden_layers,
            "hidden_size": model_config.hidden_size,
            "intermediate_size": generated_config.intermediate_size,
            "num_attention_heads": generated_config.num_attention_heads,
            "num_key_value_heads": generated_config.num_key_value_heads,
            "vocab_size": generated_config.vocab_size,
            "max_position_embeddings": generated_config.max_position_embeddings,
            "seed": SEED,
        },
        "workload": {
            "batch_size": BATCH_SIZE,
            "input_len": INPUT_LEN,
            "useful_tokens": useful_tokens,
            "useful_layer_tokens": useful_layer_tokens,
            "warm_iters": args.warm_iters,
            "input_ids_min": int(input_ids.min().item()),
            "input_ids_max": int(input_ids.max().item()),
        },
        "numerical_gates": {
            "sglang_normal_vs_split": {
                "atol": INTERNAL_ATOL,
                "rtol": INTERNAL_RTOL,
                "unchanged_from_existing_split_prefill_test": True,
            },
            "independent_torch_control": {
                "atol": CONTROL_ATOL,
                "rtol": CONTROL_RTOL,
                "purpose": "cross-framework bf16 control; separate from the unchanged internal gate",
            },
        },
        "timing_method": "torch.cuda.synchronize before and after each complete bounded forward; time.perf_counter",
        "reproduction_command": "PYTHONPATH=/job/sglang/python /opt/venv/bin/python reports/j-c905c499f1be/chunk_partition_model_runner.py --warm-iters 5",
        "memory_method": "torch.cuda.reset_peak_memory_stats before each case; max_memory_allocated and max_memory_reserved after cold plus bounded warm forwards",
        "cases": results,
        "independent_torch_control": {
            "logits_vs_normal": control_logits_metrics,
            "final_hidden_states_vs_normal": control_hidden_metrics,
        },
        "paths": {
            "python": sys.executable,
            "torch": torch.__file__,
            "transformers": transformers.__file__,
            "sglang": sglang.__file__,
            "model_runner_source": model_runner_module.__file__,
            "benchmark_script": str(Path(__file__).resolve()),
            "sgl_kernel": sgl_kernel.__file__,
            "triton": triton.__file__,
        },
        "runtime": {
            "attention_backend": server_args.attention_backend,
            "cuda_graph_disabled": server_args.disable_cuda_graph,
            "mem_fraction_static": server_args.mem_fraction_static,
            "tp_size": server_args.tp_size,
        },
        "versions": {
            "torch": torch.__version__,
            "hip": getattr(torch.version, "hip", None),
            "transformers": transformers.__version__,
        },
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "device_count": torch.cuda.device_count(),
        },
        "image_identity": {
            "operator_provided_local_image_id": "sha256:dfc9419089c338b5712da4841768b38b1ab79f3da41f8c58c3cd4dfcc1147ff1",
            "note": "operator-provided local image ID; not treated as a pullable registry digest",
        },
        "limits": {
            "wall_clock_seconds": 7200,
            "max_workload_cases": 6,
            "max_warm_iterations_per_case": 10,
            "weight_limit_bytes": 4 * 1024**3,
            "no_checkpoint_download": True,
            "no_tokenizer_download": True,
            "no_synthetic_burn": True,
        },
    }
    output_path = Path(__file__).with_name("results.json")
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
