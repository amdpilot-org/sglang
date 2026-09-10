#!/usr/bin/env python3
"""Reproduce and control-check the Qwen4-Exp PLE weight-scale load path."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from sglang.srt.configs.qwen4_exp import Qwen4ExpConfig
from sglang.srt.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.models.qwen4_exp import Qwen4ExpForConditionalGeneration
from sglang.srt.runtime_context import get_context, get_parallel


CHECKPOINT_SCALE_NAME = (
    "model.layers.1.ple.ple_embedding.ngram_embedding.weight_scale"
)
CHECKPOINT_SHARD_NAME = (
    "model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight"
)
UNRELATED_SCALE_NAME = (
    "model.layers.1.ple.ple_embedding.ngram_embedding.unrelated_scale"
)
MODEL_SCALE_NAME = (
    "model.layers.0.ple.ple_embedding.ngram_embedding.weight_scale"
)
MODEL_SHARD_NAME = (
    "model.layers.0.ple.ple_embedding.ngram_embedding.shard_0.weight"
)
SCALE_VALUE = 0.00019931793212890625


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--source-label", required=True)
    parser.add_argument("--require-commit", required=True)
    return parser.parse_args()


def git(source_root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(source_root), *args], text=True
    ).strip()


def verify_source(args: argparse.Namespace) -> dict[str, Any]:
    if not args.source_root.is_dir():
        raise RuntimeError(f"source root is unavailable: {args.source_root}")
    if not (args.source_root / ".git").exists():
        raise RuntimeError(f"source root is not a git worktree: {args.source_root}")

    commit = git(args.source_root, "rev-parse", "HEAD")
    if commit != args.require_commit:
        raise RuntimeError(
            "source branch commit mismatch: "
            f"expected {args.require_commit}, got {commit}"
        )

    import sglang

    sglang_path = Path(sglang.__file__).resolve()
    expected_prefix = (args.source_root / "python").resolve()
    if expected_prefix not in sglang_path.parents:
        raise RuntimeError(
            f"imported sglang is not from {expected_prefix}: {sglang_path}"
        )

    return {
        "label": args.source_label,
        "root": str(args.source_root.resolve()),
        "commit": commit,
        "branch": git(args.source_root, "branch", "--show-current"),
        "python": os.sys.executable,
        "sglang_path": str(sglang_path),
        "qwen4_exp_path": str(
            (expected_prefix / "sglang/srt/models/qwen4_exp.py").resolve()
        ),
    }


def reserve_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def make_fixture(path: Path) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(36616)
    weight = torch.randn((64, 8), generator=generator, dtype=torch.float32)
    weight = (weight * 0.25).clamp(-1.0, 1.0).to(torch.float8_e4m3fn)
    scale = torch.tensor([SCALE_VALUE], dtype=torch.bfloat16)
    unrelated_unit_scale = torch.ones(1, dtype=torch.bfloat16)

    tensors = {
        CHECKPOINT_SHARD_NAME: weight,
        CHECKPOINT_SCALE_NAME: scale,
        UNRELATED_SCALE_NAME: unrelated_unit_scale,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path))
    return tensors


def make_model() -> Qwen4ExpForConditionalGeneration:
    config = Qwen4ExpConfig(
        language_model_only=True,
        text_config={
            "hidden_size": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "intermediate_size": 32,
            "mamba_intermediate_size": 16,
            "mamba_num_heads": 2,
            "mamba_head_dim": 8,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 2,
            "linear_key_head_dim": 8,
            "linear_value_head_dim": 8,
            "linear_conv_kernel_dim": 4,
            "num_experts": 2,
            "num_experts_per_tok": 1,
            "moe_intermediate_size": 8,
            "shared_expert_intermediate_size": 8,
            "vocab_size": 32,
            "rms_norm_eps": 1e-6,
            "eos_token_id": 31,
            "layer_types": ["linear_attention", "linear_attention"],
            "ple_layer_ids": [1],
            "ple_embed_dim": 16,
            "ngram_size": 2,
            "heads_per_ngram": 2,
            "ngram_vocab_size_base": 16,
            "make_ngram_vocab_size_divisible_by": 8,
            "ple_embedding_dtype": "float8_e4m3fn",
            "split_ngram_parts": 1,
            "indexer_n_heads": 2,
            "indexer_kv_heads": 1,
            "indexer_head_dim": 8,
            "indexer_budget": 4,
            "indexer_compress_ratio": 2,
        },
    )
    torch.set_default_dtype(torch.bfloat16)
    model = Qwen4ExpForConditionalGeneration(config).cuda()
    ple = model.model.layers[0].ple.ple_embedding
    return model


def reproduce_assertion(
    model: Qwen4ExpForConditionalGeneration, fixture: dict[str, torch.Tensor]
) -> dict[str, Any]:
    try:
        model.load_weights(
            [
                (CHECKPOINT_SCALE_NAME, fixture[CHECKPOINT_SCALE_NAME]),
                (CHECKPOINT_SHARD_NAME, fixture[CHECKPOINT_SHARD_NAME]),
            ]
        )
    except AssertionError as error:
        return {
            "reproduced": True,
            "error": str(error),
            "scale_name": CHECKPOINT_SCALE_NAME,
            "scale_value": SCALE_VALUE,
            "consumed": False,
            "intentionally_skipped": False,
            "reason": "checkpoint layer name has no matching model buffer",
        }
    raise RuntimeError("expected the non-unit PLE weight_scale assertion")


def run_control(
    model: Qwen4ExpForConditionalGeneration, fixture: dict[str, torch.Tensor]
) -> dict[str, Any]:
    model.load_weights(
        [
            (MODEL_SCALE_NAME, fixture[CHECKPOINT_SCALE_NAME]),
            (MODEL_SHARD_NAME, fixture[CHECKPOINT_SHARD_NAME]),
        ]
    )
    ple = model.model.layers[0].ple.ple_embedding
    loaded_weight = ple.ngram_embedding.weight.detach().cpu()
    loaded_scale = ple.ngram_embedding.weight_scale.detach().cpu()
    torch.testing.assert_close(
        loaded_scale, fixture[CHECKPOINT_SCALE_NAME], rtol=0, atol=0
    )
    original_rows = int(ple.ngram_embedding.org_vocab_size)
    torch.testing.assert_close(
        loaded_weight[:original_rows],
        fixture[CHECKPOINT_SHARD_NAME][:original_rows].to(loaded_weight.dtype),
        rtol=0,
        atol=0,
    )

    lookup_ids = torch.tensor([0, 7, 39], device="cuda")
    embedding = ple.ngram_embedding(lookup_ids)
    scaled_embedding = embedding * ple.ngram_embedding.weight_scale
    direct_reference = (
        fixture[CHECKPOINT_SHARD_NAME]
        .to(torch.bfloat16)[lookup_ids.cpu()]
        .to(torch.float32)
        .to("cuda")
        * fixture[CHECKPOINT_SCALE_NAME].to(torch.float32).to("cuda")
    )
    torch.testing.assert_close(
        scaled_embedding,
        direct_reference.to(torch.bfloat16),
        rtol=0,
        atol=0,
    )

    return {
        "scale_name": MODEL_SCALE_NAME,
        "scale_value": SCALE_VALUE,
        "consumed": True,
        "intentionally_skipped": False,
        "embedding_ids": lookup_ids.cpu().tolist(),
        "embedding_dtype": str(embedding.dtype),
        "weight_dtype": str(loaded_weight.dtype),
        "original_vocab_rows": original_rows,
        "max_abs_difference": float(
            (scaled_embedding.float() - direct_reference).abs().max().item()
        ),
    }


def check_intentional_skip(
    model: Qwen4ExpForConditionalGeneration, fixture: dict[str, torch.Tensor]
) -> dict[str, Any]:
    model.load_weights([(UNRELATED_SCALE_NAME, fixture[UNRELATED_SCALE_NAME])])
    return {
        "scale_name": UNRELATED_SCALE_NAME,
        "scale_value": 1.0,
        "consumed": False,
        "intentionally_skipped": True,
        "reason": "unrecognized _scale is allowed to be skipped only when unit-valued",
    }


def main() -> None:
    args = parse_args()
    source = verify_source(args)
    fixture_tensors = make_fixture(args.fixture)
    fixture = load_file(str(args.fixture))

    port = reserve_port()
    os.environ.update(
        RANK="0",
        WORLD_SIZE="1",
        LOCAL_RANK="0",
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
    )
    torch.cuda.set_device(0)
    init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method="env://",
        backend="gloo",
    )
    initialize_model_parallel(tensor_model_parallel_size=1)
    override = get_context().override_server_args(tp_size=1)
    override.install()
    try:
        with get_parallel().override(
            attn_tp_rank=0, attn_tp_size=1, attn_dp_size=1, tp_size=1
        ):
            reproduction_model = make_model()
            reproduction = reproduce_assertion(reproduction_model, fixture)
            control_model = make_model()
            control = run_control(control_model, fixture)
            skip_model = make_model()
            intentional_skip = check_intentional_skip(skip_model, fixture)
    finally:
        override.restore()

    result = {
        "source": source,
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "device_count": torch.cuda.device_count(),
            "capability": list(torch.cuda.get_device_capability(0)),
        },
        "torch_version": torch.__version__,
        "fixture": {
            "path": str(args.fixture.resolve()),
            "tensor_names": sorted(fixture_tensors),
            "bytes": args.fixture.stat().st_size,
        },
        "reproduction": reproduction,
        "mapped_control": control,
        "intentional_skip": intentional_skip,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
