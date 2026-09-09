from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="SGLang repository root")
    parser.add_argument("--label", required=True)
    parser.add_argument("--budget-gib", type=float, default=1.0)
    parser.add_argument("--context-len", type=int, default=131072)
    parser.add_argument("--num-attention-heads", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repository = Path(args.source).resolve()
    sys.path.insert(0, str(repository / "python"))

    import torch
    from sglang.srt.configs.model_config import AttentionArch
    from sglang.srt.layers.attention import aiter_backend
    from sglang.srt.model_executor import model_runner_kv_cache_mixin

    aiter_backend.get_attention_tp_size = lambda: 1
    model_runner_kv_cache_mixin.get_attention_tp_size = lambda: 1

    torch.cuda.init()
    device = "cuda"
    properties = torch.cuda.get_device_properties(0)
    server_args = SimpleNamespace(
        max_running_requests=None,
        max_total_tokens=None,
        page_size=1,
        speculative_num_draft_tokens=None,
        speculative_num_steps=None,
    )
    model_config = SimpleNamespace(
        attention_arch=AttentionArch.MHA,
        context_len=args.context_len,
        dtype=torch.bfloat16,
        get_num_kv_heads=lambda tp_size: args.num_kv_heads,
        head_dim=args.head_dim,
        is_hybrid_swa=False,
        is_multimodal=False,
        num_attention_heads=args.num_attention_heads,
        v_head_dim=args.head_dim,
    )
    runner = SimpleNamespace(
        device=device,
        gpu_id=0,
        dp_size=1,
        pp_size=1,
        is_draft_worker=False,
        kv_cache_dtype=torch.bfloat16,
        mambaish_config=None,
        mem_fraction_static=1.0,
        model_config=model_config,
        num_effective_layers=args.num_layers,
        server_args=server_args,
        spec_algorithm=SimpleNamespace(is_none=lambda: True),
        start_layer=0,
        end_layer=args.num_layers,
        use_mla_backend=False,
    )

    cell_size = model_runner_kv_cache_mixin.ModelRunnerKVCacheMixin.get_cell_size_per_token(
        runner, args.num_layers
    )
    budget_bytes = int(args.budget_gib * (1 << 30))
    profiled_tokens = budget_bytes // cell_size
    max_total_num_tokens = model_runner_kv_cache_mixin.ModelRunnerKVCacheMixin._resolve_token_capacity(
        runner, profiled_tokens
    )
    max_running_requests = model_runner_kv_cache_mixin.ModelRunnerKVCacheMixin._resolve_max_num_reqs(
        runner, max_total_num_tokens
    )

    req_to_token_pool = SimpleNamespace(
        req_to_token=torch.empty(
            (max_running_requests, args.context_len),
            dtype=torch.int32,
            device=device,
        ),
        size=max_running_requests,
    )
    token_to_kv_pool = SimpleNamespace(
        get_value_buffer=lambda layer_id: torch.empty(
            (1, args.num_kv_heads, args.head_dim),
            dtype=torch.bfloat16,
            device=device,
        )
    )
    runner.device = device
    runner.kv_cache_dtype = torch.bfloat16
    runner.max_total_num_tokens = max_total_num_tokens
    runner.model_config = model_config
    runner.req_to_token_pool = req_to_token_pool
    runner.server_args = server_args
    runner.token_to_kv_pool = token_to_kv_pool

    torch.cuda.empty_cache()
    allocated_before = int(torch.cuda.memory_allocated(device))
    backend = aiter_backend.AiterAttnBackend(runner, skip_prefill=True)
    torch.cuda.synchronize(device)
    allocated_after = int(torch.cuda.memory_allocated(device))

    workspace_bytes = int(backend.workspace_buffer.numel())
    predicted_workspace_bytes = int(
        max_running_requests
        * args.num_attention_heads
        * backend.max_num_partitions
        * args.head_dim
        * 4
        + 2
        * max_running_requests
        * args.num_attention_heads
        * backend.max_num_partitions
        * 4
    )
    kv_bytes = max_total_num_tokens * cell_size
    headroom_before_cache = budget_bytes - kv_bytes - workspace_bytes

    result = {
        "label": args.label,
        "source": str(repository / "python"),
        "git_commit": subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
        ).strip(),
        "gpu": {
            "name": properties.name,
            "total_bytes": int(properties.total_memory),
            "gcn_arch": getattr(properties, "gcn_arch_name", None),
        },
        "budget_bytes": budget_bytes,
        "context_len": args.context_len,
        "num_attention_heads": args.num_attention_heads,
        "num_kv_heads": args.num_kv_heads,
        "head_dim": args.head_dim,
        "num_layers": args.num_layers,
        "cell_size_bytes": cell_size,
        "max_total_num_tokens": max_total_num_tokens,
        "max_running_requests": max_running_requests,
        "max_num_partitions": backend.max_num_partitions,
        "predicted_workspace_bytes": predicted_workspace_bytes,
        "allocated_workspace_bytes": workspace_bytes,
        "backend_allocated_delta_bytes": allocated_after - allocated_before,
        "predicted_kv_cache_bytes": kv_bytes,
        "headroom_before_cache_bytes": headroom_before_cache,
        "headroom_before_cache_gib": headroom_before_cache / (1 << 30),
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
