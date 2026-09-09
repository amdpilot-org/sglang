from __future__ import annotations

import argparse
import dataclasses
import inspect
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="SGLang python source root")
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
    source = str(repository / "python")
    sys.path.insert(0, source)

    import torch
    from sglang.srt.configs.model_config import AttentionArch
    from sglang.srt.distributed.parallel_state_wrapper import ParallelState
    from sglang.srt.mem_cache import kv_cache_configurator as kv_cache_configurator_module
    from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
    from sglang.srt.model_executor.pool_configurator import (
        create_memory_pool_configurator,
    )
    from sglang.srt.runtime_context import get_context, get_parallel
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    torch.cuda.init()
    device = "cuda"
    properties = torch.cuda.get_device_properties(0)

    model_config = SimpleNamespace(
        attention_arch=AttentionArch.MHA,
        context_len=args.context_len,
        dtype=torch.bfloat16,
        full_attention_layer_ids=list(range(args.num_layers)),
        get_num_kv_heads=lambda tp_size, dcp_size=1: args.num_kv_heads,
        head_dim=args.head_dim,
        hf_config=SimpleNamespace(
            architectures=["LlamaForCausalLM"],
            model_type="llama",
            get_text_config=lambda: None,
        ),
        is_draft_model=False,
        is_multimodal=False,
        linear_attn_registry_result=None,
        num_attention_heads=args.num_attention_heads,
        swa_attention_layer_ids=[],
        v_head_dim=args.head_dim,
    )

    fields = {
        "attention_backend": "aiter",
        "mem_fraction_static": 1.0,
        "max_running_requests": None,
        "max_total_tokens": None,
        "page_size": 1,
        "speculative_algorithm": None,
        "speculative_num_draft_tokens": None,
        "speculative_num_steps": None,
    }
    with get_context().override_server_args(**fields), get_parallel().override(
        attn_tp_size=1
    ):
        ps = ParallelState.trivial()
        configurator_kwargs = {
            "device": device,
            "gpu_id": 0,
            "ps": ps,
            "pp_group": SimpleNamespace(rank_in_group=0),
            "model_config": model_config,
            "server_args": get_context().server_args,
            "kv_cache_dtype": torch.bfloat16,
            "model_dtype": torch.bfloat16,
            "page_size": 1,
            "sliding_window_size": None,
            "spec_algorithm": SpeculativeAlgorithm.NONE,
            "is_draft_worker": False,
            "post_capture_kv_active": False,
            "spec_aux_config": SimpleNamespace(
                eagle_draft_num_layers=None,
                eagle_draft_swa_num_layers=None,
                dflash_draft_num_layers=None,
            ),
            "is_hybrid_swa": False,
            "is_hybrid_swa_compress": False,
            "use_mla_backend": False,
            "layer_info": SimpleNamespace(
                start_layer=0,
                end_layer=args.num_layers,
                num_effective_layers=args.num_layers,
            ),
            "forward_stream": None,
            "req_to_token_pool": None,
            "token_to_kv_pool_allocator": None,
            "memory_pool_config": None,
        }
        if "model" in {field.name for field in dataclasses.fields(KVCacheConfigurator)}:
            configurator_kwargs["model"] = None
        configurator = KVCacheConfigurator(**configurator_kwargs)

        budget_bytes = int(args.budget_gib * (1 << 30))
        pool_configurator = create_memory_pool_configurator(configurator)
        kv_cache_configurator_module.get_available_gpu_memory = (
            lambda *arguments, **keyword_arguments: args.budget_gib
        )
        kv_cache_configurator_module.get_world_group = lambda: SimpleNamespace(
            world_size=1, cpu_group=None
        )
        profile_signature = inspect.signature(
            configurator._profile_available_bytes
        )
        if "cell_size" in profile_signature.parameters:
            available_bytes = configurator._profile_available_bytes(
                0, cell_size=pool_configurator._cell_size
            )
        else:
            available_bytes = configurator._profile_available_bytes(0)
        pool_config = configurator.config_from_budget(available_bytes)
        max_total_num_tokens = pool_config.max_total_num_tokens
        max_running_requests = configurator.resolve_max_num_reqs(
            max_total_num_tokens
        )

        from sglang.srt.layers.attention.aiter_backend import AiterAttnBackend

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
        model_runner = SimpleNamespace(
            device=device,
            kv_cache_dtype=torch.bfloat16,
            max_total_num_tokens=max_total_num_tokens,
            model_config=model_config,
            req_to_token_pool=req_to_token_pool,
            server_args=get_context().server_args,
            token_to_kv_pool=token_to_kv_pool,
        )

        torch.cuda.empty_cache()
        allocated_before = int(torch.cuda.memory_allocated(device))
        backend = AiterAttnBackend(model_runner, skip_prefill=True)
        torch.cuda.synchronize(device)
        allocated_after = int(torch.cuda.memory_allocated(device))

        workspace = backend.workspace_buffer
        workspace_bytes = int(workspace.numel())
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
        kv_bytes = int(
            max_total_num_tokens
            * args.num_layers
            * args.num_kv_heads
            * args.head_dim
            * 2
            * 2
        )
        headroom_before_cache = budget_bytes - kv_bytes - workspace_bytes

        result = {
            "label": args.label,
            "source": source,
            "git_commit": subprocess.check_output(
                ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
            ).strip(),
            "gpu": {
                "name": properties.name,
                "total_bytes": int(properties.total_memory),
                "gcn_arch": getattr(properties, "gcn_arch_name", None),
            },
            "budget_bytes": budget_bytes,
            "planning_available_bytes": available_bytes,
            "context_len": args.context_len,
            "num_attention_heads": args.num_attention_heads,
            "num_kv_heads": args.num_kv_heads,
            "head_dim": args.head_dim,
            "num_layers": args.num_layers,
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
