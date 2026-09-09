#!/usr/bin/env python3
"""Validate GLM-5.3 transformers-layout loading without a model download."""

from __future__ import annotations

import argparse
import json
import logging
import socket
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Path to the SGLang python source tree to import",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def tensor_stats(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    difference = (actual.float() - expected.float()).abs()
    return {
        "shape": list(actual.shape),
        "max_abs_diff": float(difference.max().item()),
        "allclose": bool(torch.allclose(actual, expected, rtol=0.0, atol=0.0)),
    }


def bitwise_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    if left.dtype in (torch.float32, torch.float16, torch.bfloat16):
        return torch.equal(left.view(torch.int32), right.view(torch.int32))
    return torch.equal(left, right)


def build_fixture() -> tuple[list[tuple[str, torch.Tensor]], dict[str, torch.Tensor]]:
    generator = torch.Generator().manual_seed(5303)

    def random(shape: Iterable[int]) -> torch.Tensor:
        return torch.randn(tuple(shape), generator=generator, dtype=torch.float32)

    prefix = "model.language_model.layers.0"
    tensors = {
        "attn_hc_fn": random((8, 16)),
        "attn_hc_base": random((8,)),
        "attn_hc_scale": random((3,)),
        "ffn_hc_fn": random((8, 16)),
        "ffn_hc_base": random((8,)),
        "ffn_hc_scale": random((3,)),
        "f_a_proj": random((4, 8)),
        "f_b_proj": random((8, 4)),
        "dt_bias": random((8,)),
        "A_log": random((2,)),
        "conv1d": random((24, 1, 3)),
        "gate_up_proj": random((2, 12, 8)),
        "down_proj": random((2, 8, 6)),
    }
    fixture = [
        (f"{prefix}.attn_hc.fn", tensors["attn_hc_fn"]),
        (f"{prefix}.attn_hc.base", tensors["attn_hc_base"]),
        (f"{prefix}.attn_hc.scale", tensors["attn_hc_scale"]),
        (f"{prefix}.ffn_hc.fn", tensors["ffn_hc_fn"]),
        (f"{prefix}.ffn_hc.base", tensors["ffn_hc_base"]),
        (f"{prefix}.ffn_hc.scale", tensors["ffn_hc_scale"]),
        (f"{prefix}.self_attn.forget_gate.f_a_proj.weight", tensors["f_a_proj"]),
        (f"{prefix}.self_attn.forget_gate.f_b_proj.weight", tensors["f_b_proj"]),
        (f"{prefix}.self_attn.forget_gate.dt_bias", tensors["dt_bias"]),
        (f"{prefix}.self_attn.forget_gate.A_log", tensors["A_log"]),
        (f"{prefix}.self_attn.conv1d.weight", tensors["conv1d"]),
        (f"{prefix}.mlp.experts.gate_up_proj", tensors["gate_up_proj"]),
        (f"{prefix}.mlp.experts.down_proj", tensors["down_proj"]),
    ]
    return fixture, tensors


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    sys.path.insert(0, str(source_root))
    torch.set_grad_enabled(False)

    from sglang.srt.configs.glm5_next import Glm5NextConfig, Glm5NextTextConfig
    from sglang.srt.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    from sglang.srt.models.glm5_next import Glm5NextForConditionalGeneration
    from sglang.srt.runtime_context import get_context, get_parallel

    git_root = source_root.parent
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=git_root, text=True
    ).strip()

    text_config = Glm5NextTextConfig(
        vocab_size=17,
        hidden_size=8,
        intermediate_size=12,
        moe_intermediate_size=6,
        num_hidden_layers=1,
        num_attention_heads=1,
        n_routed_experts=2,
        num_experts_per_tok=1,
        n_shared_experts=0,
        first_k_dense_replace=0,
        q_lora_rank=None,
        kv_lora_rank=4,
        qk_nope_head_dim=4,
        qk_rope_head_dim=0,
        v_head_dim=4,
        mhc=True,
        hc_mult=2,
        linear_head_dim=4,
        linear_num_heads=2,
        linear_conv_kernel_dim=3,
        layer_types=["linear_attention"],
    )
    config = Glm5NextConfig(
        text_config=text_config,
        vision_config={
            "depth": 0,
            "hidden_size": 8,
            "num_heads": 1,
            "out_hidden_size": 8,
            "intermediate_size": 8,
            "swiglu_limit": 7.0,
        },
        language_only=True,
    )

    port = free_port()
    init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method=f"tcp://127.0.0.1:{port}",
        backend="gloo",
    )
    initialize_model_parallel(tensor_model_parallel_size=1, backend="gloo")

    parallel_fields = {
        "world_size": 1,
        "world_rank": 0,
        "tp_size": 1,
        "tp_rank": 0,
        "pp_size": 1,
        "pp_rank": 0,
        "moe_ep_size": 1,
        "moe_ep_rank": 0,
        "moe_dp_size": 1,
        "moe_dp_rank": 0,
        "moe_tp_size": 1,
        "moe_tp_rank": 0,
        "attn_tp_size": 1,
        "attn_tp_rank": 0,
        "attn_cp_size": 1,
        "attn_cp_rank": 0,
        "dcp_enabled": False,
        "dcp_size": 1,
        "dcp_rank": 0,
        "attn_dcp_size": 1,
        "attn_dcp_rank": 0,
        "attn_dp_size": 1,
        "attn_dp_rank": 0,
    }
    with get_context().override_server_args(), get_parallel().override(
        **parallel_fields
    ):
        model = Glm5NextForConditionalGeneration(config).to("cuda")
        reference = Glm5NextForConditionalGeneration(config).to("cuda")
        reference.load_state_dict(model.state_dict())

        fixture, tensors = build_fixture()
        prefix = "model.language_model.layers.0"
        reference.model.layers[0].hc_attn_fn.copy_(tensors["attn_hc_fn"].cuda())
        reference.model.layers[0].hc_attn_base.copy_(
            tensors["attn_hc_base"].cuda()
        )
        reference.model.layers[0].hc_attn_scale.copy_(
            tensors["attn_hc_scale"].cuda()
        )
        reference.model.layers[0].hc_ffn_fn.copy_(tensors["ffn_hc_fn"].cuda())
        reference.model.layers[0].hc_ffn_base.copy_(tensors["ffn_hc_base"].cuda())
        reference.model.layers[0].hc_ffn_scale.copy_(
            tensors["ffn_hc_scale"].cuda()
        )
        reference_attn = reference.model.layers[0].self_attn
        reference_attn.fused_qkvbfg_a_proj.weight[26:30].copy_(
            tensors["f_a_proj"].cuda()
        )
        reference_attn.fused_fg_b_proj.weight[0].copy_(
            tensors["f_b_proj"].cuda()
        )
        reference_attn.dt_bias.copy_(tensors["dt_bias"].cuda())
        reference_attn.A_log.view(-1).copy_(tensors["A_log"].cuda())
        reference_attn.qkv_conv1d.weight.copy_(tensors["conv1d"].cuda())
        reference.model.layers[0].mlp.experts.w13_weight.copy_(
            tensors["gate_up_proj"].cuda()
        )
        reference.model.layers[0].mlp.experts.w2_weight.copy_(
            tensors["down_proj"].cuda()
        )

        records = []
        loader_calls = []
        fused_module = model.model.layers[0].self_attn.fused_qkvbfg_a_proj
        fused_param = model.model.layers[0].self_attn.fused_qkvbfg_a_proj.weight
        original_fused_loader = fused_param.weight_loader

        def traced_fused_loader(
            param: torch.nn.Parameter,
            loaded_weight: torch.Tensor,
            loaded_shard_id: int,
        ) -> None:
            loader_calls.append(
                {
                    "shard_id": loaded_shard_id,
                    "shape": list(loaded_weight.shape),
                    "first_value": float(loaded_weight.flatten()[0].item()),
                    "computed_offset": sum(
                        fused_module.output_partition_sizes[:loaded_shard_id]
                    ),
                    "param_output_dim": int(fused_param.output_dim),
                }
            )
            original_fused_loader(param, loaded_weight, loaded_shard_id)

        fused_param.weight_loader = traced_fused_loader
        target_names = {
            f"{prefix}.attn_hc.fn": "model.layers.0.hc_attn_fn",
            f"{prefix}.attn_hc.base": "model.layers.0.hc_attn_base",
            f"{prefix}.attn_hc.scale": "model.layers.0.hc_attn_scale",
            f"{prefix}.ffn_hc.fn": "model.layers.0.hc_ffn_fn",
            f"{prefix}.ffn_hc.base": "model.layers.0.hc_ffn_base",
            f"{prefix}.ffn_hc.scale": "model.layers.0.hc_ffn_scale",
            f"{prefix}.self_attn.forget_gate.f_a_proj.weight": "model.layers.0.self_attn.fused_qkvbfg_a_proj.weight[26:30]",
            f"{prefix}.self_attn.forget_gate.f_b_proj.weight": "model.layers.0.self_attn.fused_fg_b_proj.weight[0]",
            f"{prefix}.self_attn.forget_gate.dt_bias": "model.layers.0.self_attn.dt_bias",
            f"{prefix}.self_attn.forget_gate.A_log": "model.layers.0.self_attn.A_log",
            f"{prefix}.self_attn.conv1d.weight": "model.layers.0.self_attn.qkv_conv1d.weight",
            f"{prefix}.mlp.experts.gate_up_proj": "model.layers.0.mlp.experts.w13_weight",
            f"{prefix}.mlp.experts.down_proj": "model.layers.0.mlp.experts.w2_weight",
        }

        model.load_weights(fixture)
        torch.cuda.synchronize()

        model_layer = model.model.layers[0]
        model_attn = model_layer.self_attn
        reference_layer = reference.model.layers[0]
        reference_attn = reference_layer.self_attn
        comparisons = {
            "model.layers.0.hc_attn_fn": (
                model_layer.hc_attn_fn,
                reference_layer.hc_attn_fn,
            ),
            "model.layers.0.hc_attn_base": (
                model_layer.hc_attn_base,
                reference_layer.hc_attn_base,
            ),
            "model.layers.0.hc_attn_scale": (
                model_layer.hc_attn_scale,
                reference_layer.hc_attn_scale,
            ),
            "model.layers.0.hc_ffn_fn": (
                model_layer.hc_ffn_fn,
                reference_layer.hc_ffn_fn,
            ),
            "model.layers.0.hc_ffn_base": (
                model_layer.hc_ffn_base,
                reference_layer.hc_ffn_base,
            ),
            "model.layers.0.hc_ffn_scale": (
                model_layer.hc_ffn_scale,
                reference_layer.hc_ffn_scale,
            ),
            "model.layers.0.self_attn.fused_qkvbfg_a_proj.weight[26:30]": (
                model_attn.fused_qkvbfg_a_proj.weight[26:30],
                reference_attn.fused_qkvbfg_a_proj.weight[26:30],
            ),
            "model.layers.0.self_attn.fused_fg_b_proj.weight[0]": (
                model_attn.fused_fg_b_proj.weight[0],
                reference_attn.fused_fg_b_proj.weight[0],
            ),
            "model.layers.0.self_attn.dt_bias": (
                model_attn.dt_bias,
                reference_attn.dt_bias,
            ),
            "model.layers.0.self_attn.A_log": (
                model_attn.A_log,
                reference_attn.A_log,
            ),
            "model.layers.0.self_attn.qkv_conv1d.weight": (
                model_attn.qkv_conv1d.weight,
                reference_attn.qkv_conv1d.weight,
            ),
            "model.layers.0.mlp.experts.w13_weight": (
                model_layer.mlp.experts.w13_weight,
                reference_layer.mlp.experts.w13_weight,
            ),
            "model.layers.0.mlp.experts.w2_weight": (
                model_layer.mlp.experts.w2_weight,
                reference_layer.mlp.experts.w2_weight,
            ),
        }
        for fixture_name, target_name in target_names.items():
            stats = tensor_stats(*comparisons[target_name])
            records.append(
                {
                    "fixture_key": fixture_name,
                    "target": target_name,
                    **stats,
                }
            )

        unexpected_key = "model.language_model.layers.0.unexpected.weight"
        before_unexpected = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
        }
        model.load_weights([(unexpected_key, torch.zeros(8))])
        unexpected_changed_names = [
            name
            for name, parameter in model.named_parameters()
            if not bitwise_equal(before_unexpected[name], parameter.detach())
        ]
        unexpected_changed = bool(unexpected_changed_names)
        missing_target_unchanged = bitwise_equal(
            model.model.layers[0].hc_ffn_base,
            torch.full_like(model.model.layers[0].hc_ffn_base, -5303.0),
        )
        missing_key = "model.language_model.layers.0.ffn_hc.base"
        model.model.layers[0].hc_ffn_base.fill_(-5303.0)
        missing_fixture = [item for item in fixture if item[0] != missing_key]
        model.load_weights(missing_fixture)
        missing_target_unchanged = bitwise_equal(
            model.model.layers[0].hc_ffn_base,
            torch.full_like(model.model.layers[0].hc_ffn_base, -5303.0),
        )

    result = {
        "source_root": str(source_root),
        "source_commit": commit,
        "glm_module": str(sys.modules[Glm5NextForConditionalGeneration.__module__].__file__),
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "gcn_arch_name": getattr(
                torch.cuda.get_device_properties(0), "gcnArchName", ""
            ),
            "device_count": torch.cuda.device_count(),
        },
        "torch_version": torch.__version__,
        "hip_version": torch.version.hip,
        "fixture_tensor_count": len(fixture),
        "fused_qkvbfg_output_partition_sizes": list(
            model.model.layers[0]
            .self_attn.fused_qkvbfg_a_proj.output_partition_sizes
        ),
        "fused_qkvbfg_loader_calls": loader_calls,
        "per_tensor_results": records,
        "all_intended_tensors_match": all(record["allclose"] for record in records),
        "visibility": {
            "unexpected_key": unexpected_key,
            "unexpected_changed_any_parameter": unexpected_changed,
            "unexpected_changed_parameters": unexpected_changed_names,
            "missing_key": missing_key,
            "missing_target": "model.layers.0.hc_ffn_base",
            "missing_target_unchanged": missing_target_unchanged,
            "loader_emitted_warning_or_error": False,
        },
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
