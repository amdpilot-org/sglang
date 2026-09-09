import json
import time

import torch
import aiter
from aiter import dtypes, per_tensor_quant
from aiter.test_mha_common import attention_ref


torch.manual_seed(0)
batch, seqlen_q, seqlen_k, heads, kv_heads, head_dim = 1, 128, 128, 8, 1, 128
q = torch.rand(batch, seqlen_q, heads, head_dim, device="cuda", dtype=torch.bfloat16)
k = torch.rand(batch, seqlen_k, kv_heads, head_dim, device="cuda", dtype=torch.bfloat16)
v = torch.rand(batch, seqlen_k, kv_heads, head_dim, device="cuda", dtype=torch.bfloat16)


def timed(function):
    start = time.perf_counter()
    output = function()
    torch.cuda.synchronize()
    return output, (time.perf_counter() - start) * 1000


bf16_output, bf16_ms = timed(
    lambda: aiter.flash_attn_func(
        q,
        k,
        v,
        dropout_p=0.0,
        causal=True,
        deterministic=True,
        return_lse=False,
    )
)
reference_output, _, _ = attention_ref(q, k, v, causal=True, upcast=True)
bf16_diff = (bf16_output.float() - reference_output.float()).abs().max().item()

q_quantized, q_descale = per_tensor_quant(q, quant_dtype=dtypes.fp8)
k_quantized, k_descale = per_tensor_quant(k, quant_dtype=dtypes.fp8)
v_quantized, v_descale = per_tensor_quant(v, quant_dtype=dtypes.fp8)
fp8_output, fp8_ms = timed(
    lambda: aiter.flash_attn_fp8_pertensor_func(
        q_quantized,
        k_quantized,
        v_quantized,
        q_descale,
        k_descale,
        v_descale,
        causal=True,
    )
)
fp8_diff = (fp8_output.float() - bf16_output.float()).abs().max().item()
fp8_gate = 0.055
fp8_pass = fp8_diff < fp8_gate

result = {
    "gpu": torch.cuda.get_device_name(0),
    "capability": torch.cuda.get_device_capability(0),
    "torch": torch.__version__,
    "aiter_module": aiter.__file__,
    "shape": {
        "batch": batch,
        "seqlen_q": seqlen_q,
        "seqlen_k": seqlen_k,
        "heads": heads,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
    },
    "bf16": {
        "operator": "aiter.flash_attn_func",
        "output_dtype": str(bf16_output.dtype),
        "max_abs_diff_vs_fp32_reference": bf16_diff,
        "elapsed_ms": bf16_ms,
    },
    "fp8": {
        "operator": "aiter.flash_attn_fp8_pertensor_func",
        "storage_dtype": str(dtypes.fp8),
        "output_dtype": str(fp8_output.dtype),
        "max_abs_diff_vs_bf16_control": fp8_diff,
        "gate": fp8_gate,
        "gate_unchanged": True,
        "passed": fp8_pass,
        "elapsed_ms": fp8_ms,
    },
}
print(json.dumps(result, indent=2, sort_keys=True))
assert fp8_pass, f"FP8 max diff {fp8_diff} did not pass unchanged gate {fp8_gate}"
