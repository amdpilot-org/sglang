import json
from types import SimpleNamespace

import torch
from sglang.srt.arg_groups.attention_hook import handle_attention_backend_compatibility
from sglang.srt.arg_groups.overrides import resolved_view
from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.server_args import ServerArgs


def make_args(backend, dtype, arch):
    args = ServerArgs(model_path="dummy")
    args.attention_backend = backend
    args.prefill_attention_backend = None
    args.decode_attention_backend = None
    args.kv_cache_dtype = dtype
    args._model_config = SimpleNamespace(
        attention_arch=arch,
        hf_config=SimpleNamespace(
            architectures=[
                "LlamaForCausalLM"
                if arch == AttentionArch.MHA
                else "DeepseekV3ForCausalLM"
            ]
        ),
        context_len=128,
        is_encoder_decoder=False,
        has_asymmetric_kv=False,
        has_attention_sinks=False,
        get_num_kv_heads=lambda tp_size: 8,
    )
    return args


def run_case(name, backend, dtype, arch):
    args = make_args(backend, dtype, arch)
    try:
        handle_attention_backend_compatibility(args)
        view = resolved_view(args)
        result = {
            "case": name,
            "requested_backend": backend,
            "kv_cache_dtype": dtype,
            "resolution": "admitted",
            "resolved_attention_backend": view.attention_backend,
            "resolved_prefill_backend": view.prefill_attention_backend,
            "resolved_decode_backend": view.decode_attention_backend,
        }
        if backend == "fa3":
            try:
                from sgl_kernel.flash_attn import (  # noqa: F401
                    flash_attn_varlen_func,
                    flash_attn_with_kvcache,
                    get_scheduler_metadata,
                )

                result["native_import"] = "succeeded"
            except BaseException as exc:
                result.update(
                    {
                        "native_import": "failed",
                        "native_error_type": type(exc).__name__,
                        "native_error": str(exc),
                    }
                )
        return result
    except BaseException as exc:
        traceback = exc.__traceback__
        frames = []
        while traceback is not None:
            frame = traceback.tb_frame
            frames.append(
                {
                    "file": frame.f_code.co_filename,
                    "line": traceback.tb_lineno,
                    "function": frame.f_code.co_name,
                }
            )
            traceback = traceback.tb_next
        return {
            "case": name,
            "requested_backend": backend,
            "kv_cache_dtype": dtype,
            "resolution": "rejected",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": frames,
        }


cases = []
for dtype in ("bfloat16", "fp8_e4m3"):
    cases.append(run_case(f"default_mha_{dtype}", None, dtype, AttentionArch.MHA))
    cases.append(run_case(f"explicit_aiter_{dtype}", "aiter", dtype, AttentionArch.MHA))
    cases.append(run_case(f"explicit_fa3_{dtype}", "fa3", dtype, AttentionArch.MHA))
    cases.append(
        run_case(
            f"explicit_trtllm_mla_{dtype}", "trtllm_mla", dtype, AttentionArch.MLA
        )
    )
    cases.append(
        run_case(
            f"explicit_tokenspeed_mla_{dtype}",
            "tokenspeed_mla",
            dtype,
            AttentionArch.MLA,
        )
    )
    cases.append(
        run_case(
            f"explicit_cutedsl_mla_{dtype}", "cutedsl_mla", dtype, AttentionArch.MLA
        )
    )
    cases.append(
        run_case(
            f"explicit_trtllm_mha_{dtype}", "trtllm_mha", dtype, AttentionArch.MHA
        )
    )

print(
    json.dumps(
        {
            "gpu": torch.cuda.get_device_name(0),
            "capability": torch.cuda.get_device_capability(0),
            "source": "/job/sglang",
            "cases": cases,
        },
        indent=2,
        sort_keys=True,
    )
)
