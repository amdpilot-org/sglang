import json
from types import SimpleNamespace

import torch
from sglang.srt.arg_groups.overrides import ResolvedView, _dsa_split_backend_resolution


def view(dtype, prefill=None, decode=None):
    hf_config = SimpleNamespace(
        architectures=["DeepseekV32ForCausalLM"],
        index_topk=2048,
        learnable_sink=False,
    )
    return ResolvedView(
        SimpleNamespace(
            _model_config=SimpleNamespace(hf_config=hf_config),
            kv_cache_dtype=dtype,
            dsa_prefill_backend=prefill,
            dsa_decode_backend=decode,
            enable_hisparse=False,
        )
    )


cases = []
for dtype in ("bfloat16", "fp8_e4m3"):
    for prefill, decode in (
        (None, None),
        ("tilelang", None),
        ("flashmla_sparse", None),
        (None, "tilelang"),
    ):
        result = _dsa_split_backend_resolution(view(dtype, prefill, decode))
        cases.append(
            {
                "kv_cache_dtype": dtype,
                "input_prefill": prefill,
                "input_decode": decode,
                "declared": result,
            }
        )

print(
    json.dumps(
        {
            "gpu": torch.cuda.get_device_name(0),
            "capability": torch.cuda.get_device_capability(0),
            "cases": cases,
        },
        indent=2,
    )
)
