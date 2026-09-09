import argparse
import importlib
import importlib.util
import json
import os
import subprocess
import shutil
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import torch
import triton


def install_ray_stub():
    ray = ModuleType("ray")
    ray.remote = lambda *args, **kwargs: lambda cls: cls
    experimental = ModuleType("ray.experimental")
    tqdm_ray = ModuleType("ray.experimental.tqdm_ray")
    tqdm_ray.tqdm = lambda iterable=None, *args, **kwargs: iterable
    experimental.tqdm_ray = tqdm_ray
    ray.experimental = experimental
    sys.modules["ray"] = ray
    sys.modules["ray.experimental"] = experimental
    sys.modules["ray.experimental.tqdm_ray"] = tqdm_ray


def load_tuner_module(repo):
    benchmark_dir = repo / "benchmark/kernels/fused_moe_triton"
    sys.path.insert(0, str(repo / "python"))
    sys.path.insert(0, str(benchmark_dir))
    spec = importlib.util.spec_from_file_location(
        "tuning_fused_moe_triton", benchmark_dir / "tuning_fused_moe_triton.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def quantize_asymmetric(weight, group_size):
    quantized = weight.T.contiguous()
    size_k, size_n = quantized.shape
    if group_size < size_k:
        raise NotImplementedError("This probe intentionally uses one full-width group")
    max_value = quantized.max(dim=0, keepdim=True).values
    min_value = quantized.min(dim=0, keepdim=True).values
    scale = ((max_value - min_value) / 15.0).clamp(min=1e-5)
    zero_point = torch.round(torch.abs(min_value / scale)).clamp(0, 15).to(torch.uint8)
    packed_int = torch.round(quantized / scale).to(torch.int32) + zero_point.to(torch.int32)
    packed_int = packed_int.clamp(0, 15)
    dequantized = ((packed_int - zero_point.to(torch.int32)) * scale).to(weight.dtype)
    return (
        dequantized.T.contiguous(),
        packed_int.T.contiguous().to(torch.uint8),
        scale.T.contiguous().to(weight.dtype),
        zero_point.T.contiguous(),
    )


def pack_int4(values):
    return values[:, 1::2].to(torch.uint8) * 16 + values[:, ::2].to(torch.uint8)


def pack_int4_rows(values):
    return values[1::2, :].to(torch.uint8) * 16 + values[::2, :].to(torch.uint8)


def independent_reference(hidden_states, w1_reference, w2_reference, topk_output):
    topk_weights = topk_output.topk_weights
    topk_ids = topk_output.topk_ids
    reference = torch.zeros_like(hidden_states)
    for token in range(hidden_states.shape[0]):
        for selection in range(topk_ids.shape[1]):
            expert = int(topk_ids[token, selection].item())
            router_weight = float(topk_weights[token, selection].item())
            gate_up = hidden_states[token].to(torch.float32) @ w1_reference[expert].T.to(torch.float32)
            gate = torch.nn.functional.silu(gate_up[: gate_up.shape[0] // 2])
            up = gate_up[gate_up.shape[0] // 2 :]
            expert_output = (gate * up) @ w2_reference[expert].T.to(torch.float32)
            reference[token] += router_weight * expert_output.to(reference.dtype)
    return reference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--server-model", type=Path, required=True)
    parser.add_argument("--search-space", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()

    repo = args.repo.resolve()
    model_dir = args.model.resolve()
    server_model_dir = args.server_model.resolve()
    output_dir = args.output_dir.resolve()
    config_dir = args.config_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["SGLANG_MOE_CONFIG_DIR"] = str(config_dir)
    os.environ.setdefault("TRITON_CACHE_DIR", "/job/workdir/.triton_cache")

    install_ray_stub()
    tuner = load_tuner_module(repo)
    from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig
    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import fused_moe
    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config import (
        get_config_dtype_str,
        get_config_file_name,
        get_default_config,
        get_moe_configs,
        try_get_optimal_moe_config,
    )
    from sglang.srt.layers.moe.topk import TopKConfig, select_experts
    from sglang.srt.runtime_context import get_context
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
    from sglang.srt.utils import get_device_name

    torch.set_default_device("cuda")
    torch.manual_seed(20260909)
    server_args = ServerArgs(model_path=str(server_model_dir), tp_size=1, ep_size=1)
    set_global_server_args_for_scheduler(server_args)

    model_config = tuner.get_model_config(str(model_dir), 1, 1)
    num_experts = model_config["num_experts"]
    shard_intermediate_size = model_config["shard_intermediate_size"]
    hidden_size = model_config["hidden_size"]
    topk = model_config["topk"]
    dtype = model_config["dtype"]
    block_shape = model_config["block_shape"]
    dtype_str = get_config_dtype_str(dtype, use_int4_w4a16=True)
    runtime_n = shard_intermediate_size // 2

    tuner_filename = tuner.get_config_filename(
        num_experts,
        shard_intermediate_size,
        hidden_size,
        topk,
        dtype,
        False,
        False,
        False,
        True,
        False,
        block_shape,
    )
    runtime_filename = get_config_file_name(
        num_experts, runtime_n, dtype_str, block_shape, False
    )

    search_space = json.loads(args.search_space.read_text())
    if len(search_space) != 1:
        raise ValueError("This investigation intentionally uses exactly one config")
    tuner.args = SimpleNamespace(model=str(model_dir))
    tuner_config = search_space[0]
    kernel_time_us = tuner.benchmark_config(
        tuner_config,
        3,
        num_experts,
        shard_intermediate_size,
        hidden_size,
        topk,
        dtype,
        False,
        False,
        False,
        True,
        False,
        block_shape,
        num_iters=10,
    )
    tuner_path = output_dir / tuner_filename
    tuner.save_configs({3: tuner_config}, str(tuner_path))

    version_dir = f"triton_{triton.__version__.replace('.', '_')}"
    runtime_root = config_dir / "configs" / version_dir
    runtime_root.mkdir(parents=True, exist_ok=True)
    runtime_path = runtime_root / runtime_filename
    get_moe_configs.cache_clear()
    with get_context().override_server_args(enable_deterministic_inference=False):
        pre_copy_configs = get_moe_configs(
            num_experts,
            runtime_n,
            dtype_str,
            block_shape[0],
            block_shape[1],
            per_channel_quant=False,
        )
    pre_copy_selected = None
    if pre_copy_configs is None:
        pre_copy_selected = get_default_config(
            3,
            num_experts,
            runtime_n,
            hidden_size,
            topk,
            dtype_str,
            False,
            block_shape,
        )
    else:
        pre_copy_selected = pre_copy_configs[min(pre_copy_configs, key=lambda key: abs(key - 3))]

    shutil.copyfile(tuner_path, runtime_path)
    get_moe_configs.cache_clear()
    with get_context().override_server_args(enable_deterministic_inference=False):
        runtime_configs = get_moe_configs(
            num_experts,
            runtime_n,
            dtype_str,
            block_shape[0],
            block_shape[1],
            per_channel_quant=False,
        )
    if runtime_configs is None:
        raise RuntimeError("Runtime did not select the copied tuner config")
    selected_config = runtime_configs[min(runtime_configs, key=lambda key: abs(key - 3))]

    w1_reference = torch.empty(
        (num_experts, shard_intermediate_size, hidden_size), dtype=dtype, device="cuda"
    )
    w2_reference = torch.empty(
        (num_experts, hidden_size, runtime_n), dtype=dtype, device="cuda"
    )
    w1_qweight = torch.empty(
        (num_experts, shard_intermediate_size, hidden_size // 2), dtype=torch.uint8, device="cuda"
    )
    w2_qweight = torch.empty(
        (num_experts, hidden_size, runtime_n // 2), dtype=torch.uint8, device="cuda"
    )
    w1_scale = torch.empty(
        (num_experts, shard_intermediate_size, 1), dtype=dtype, device="cuda"
    )
    w2_scale = torch.empty((num_experts, hidden_size, 1), dtype=dtype, device="cuda")
    w1_zero_point = torch.empty(
        (num_experts, shard_intermediate_size // 2, 1), dtype=torch.uint8, device="cuda"
    )
    w2_zero_point = torch.empty(
        (num_experts, hidden_size // 2, 1), dtype=torch.uint8, device="cuda"
    )

    for expert in range(num_experts):
        w1 = torch.randn(
            (shard_intermediate_size, hidden_size), dtype=dtype, device="cuda"
        ) / 10.0
        w2 = torch.randn((hidden_size, runtime_n), dtype=dtype, device="cuda") / 10.0
        w1_dequant, w1_quant, w1_scale_one, w1_zp_one = quantize_asymmetric(w1, hidden_size)
        w2_dequant, w2_quant, w2_scale_one, w2_zp_one = quantize_asymmetric(w2, runtime_n)
        w1_reference[expert] = w1_dequant
        w2_reference[expert] = w2_dequant
        w1_qweight[expert] = pack_int4(w1_quant)
        w2_qweight[expert] = pack_int4(w2_quant)
        w1_scale[expert] = w1_scale_one
        w2_scale[expert] = w2_scale_one
        w1_zero_point[expert] = pack_int4_rows(w1_zp_one)
        w2_zero_point[expert] = pack_int4_rows(w2_zp_one)

    hidden_states = torch.randn((3, hidden_size), dtype=dtype, device="cuda") / 10.0
    router_logits = torch.randn((3, num_experts), dtype=torch.float32, device="cuda")
    topk_output = select_experts(
        hidden_states,
        router_logits,
        TopKConfig(top_k=topk, renormalize=False),
    )
    reference_input = hidden_states.clone()
    gpu_output = fused_moe(
        hidden_states,
        w1_qweight,
        w2_qweight,
        topk_output,
        MoeRunnerConfig(),
        use_int4_w4a16=True,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        w1_zp=w1_zero_point,
        w2_zp=w2_zero_point,
        block_shape=block_shape,
    )
    reference = independent_reference(
        reference_input, w1_reference, w2_reference, topk_output
    )
    absolute_error = (gpu_output.to(torch.float32) - reference.to(torch.float32)).abs()
    max_absolute_error = float(absolute_error.max().item())
    mean_absolute_error = float(absolute_error.mean().item())
    reference_norm = float(reference.to(torch.float32).norm().item())
    gpu_norm = float(gpu_output.to(torch.float32).norm().item())
    relative_error = float(absolute_error.norm().item() / max(reference_norm, 1e-12))
    numerical_gate_passed = bool(max_absolute_error <= 2e-2)

    actual_config, down_config_pair = try_get_optimal_moe_config(
        w1_qweight.shape,
        (w2_qweight.shape[0], w2_qweight.shape[1], runtime_n),
        topk,
        dtype_str,
        3,
        block_shape=block_shape,
        per_channel_quant=False,
        return_down_config=True,
    )

    import sgl_kernel
    result = {
        "label": args.label,
        "repo": str(repo),
        "commit": subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip(),
        "python": sys.executable,
        "sglang_python_path": importlib.import_module("sglang").__file__,
        "fused_moe_path": importlib.import_module(
            "sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe"
        ).__file__,
        "config_path": importlib.import_module(
            "sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config"
        ).__file__,
        "sgl_kernel_path": sgl_kernel.__file__,
        "torch_version": torch.__version__,
        "triton_version": triton.__version__,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_capability": list(torch.cuda.get_device_capability(0)),
        "device_name_for_filename": get_device_name(),
        "model_config": {
            "num_experts": num_experts,
            "shard_intermediate_size": shard_intermediate_size,
            "runtime_n": runtime_n,
            "hidden_size": hidden_size,
            "topk": topk,
            "dtype": str(dtype),
            "block_shape": block_shape,
        },
        "tuner_filename": tuner_filename,
        "tuner_path": str(tuner_path),
        "runtime_filename": runtime_filename,
        "runtime_path": str(runtime_path),
        "filename_match": tuner_filename == runtime_filename,
        "tuner_config": tuner_config,
        "tuner_kernel_time_us": kernel_time_us,
        "pre_copy_runtime_configs": pre_copy_configs,
        "pre_copy_selected_config": pre_copy_selected,
        "selected_config": selected_config,
        "actual_runtime_config": actual_config,
        "actual_runtime_down_config": down_config_pair[0],
        "comparison": {
            "gpu_output_norm": gpu_norm,
            "reference_norm": reference_norm,
            "topk_weights": topk_output.topk_weights.detach().cpu().tolist(),
            "topk_ids": topk_output.topk_ids.detach().cpu().tolist(),
            "max_absolute_error": max_absolute_error,
            "mean_absolute_error": mean_absolute_error,
            "relative_l2_error": relative_error,
            "gate": "torch.testing.assert_close(gpu_output, reference, atol=2e-2, rtol=0)",
            "gate_passed": numerical_gate_passed,
        },
    }
    result_path = output_dir / f"{args.label}_result.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not numerical_gate_passed:
        raise AssertionError("Unchanged numerical gate failed")


if __name__ == "__main__":
    main()
