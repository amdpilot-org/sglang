# gfx942 router-gate permutation validation

## Scope

This follow-up tests whether row and expert permutations through the six model-local MoE router gate wrappers preserve the corresponding output permutation. It is separate from the already-fulfilled FP32 router-precision scope in mirror pull request 332 and upstream pull request 38719.

No production code was changed because the demonstrated result was negative: all supported wrapper cases preserved the combined row/expert permutation exactly.

## Environment

- Qualified image: `amdpilotv2/open-job-mi300:jit-config-readable-35122-260909`
- Operator-provided local image ID: `sha256:dfc9419089c338b5712da4841768b38b1ab79f3da41f8c58c3cd4dfcc1147ff1`
- GPU: one AMD Instinct MI300X, gfx942, Torch capability `(9, 4)`
- Python: `/opt/venv/bin/python` 3.10.12
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`
- HIP: `7.2.26015-fc0010cf6a`
- Persistent mirror base: `amdpilot-org/sglang` `main` commit `0084030179bfba86bfeb6d43f7997d4076329d2c`
- Branch: `amdpilot/j-d64d223e5d60`
- Imported SGLang source: `/job/sglang/python/sglang`
- Native Torch path: `/opt/venv/lib/python3.10/site-packages/torch`
- Native `sgl_kernel` path: `/opt/venv/lib/python3.10/site-packages/sgl_kernel`

## Installed-source baseline

The first GPU execution used the preinstalled source at commit `8eaffdf382e06b7dc50fc5c76cc5aad9c36bb0e4`, not the persistent checkout:

```bash
/opt/venv/bin/python -m pytest -q \
  /sgl-workspace/sglang/test/registered/kernels/ops/moe/test_moe_fused_gate.py \
  -x --no-header
```

Result: `1 failed, 8 passed` in `30.54` seconds. Eight Triton fused-gate cases passed their in-test references. The first CUDA/HIP JIT comparison failed during `hipcc` compilation with:

```text
static assertion failed due to requirement 'sizeof(unsigned int) == 8'
```

The supported neighboring control was:

```bash
/opt/venv/bin/python -m pytest -q \
  /sgl-workspace/sglang/test/registered/kernels/ops/moe/test_moe_preprocess.py \
  --no-header -p no:cacheprovider
```

Result: `3 failed, 14 passed` in `19.18957` seconds. All 14 `fused_moe_preprocess` permutation cases passed the independent `torch.sort` reference exactly. The three grouped-GEMM controls failed before execution because the default Triton config requested 98,304 bytes of shared memory against the 65,536-byte gfx942 hardware limit.

These installed-source results are recorded in `/job/baseline-first.json` and are not proof for the persistent checkout.

## Method

The persistent-checkout probe used:

- Seven rows, sixteen hidden channels, and eight experts.
- A finite adversarial BF16/FP32 input matrix with near-tie perturbations.
- A finite FP32 expert-weight matrix with near-tie perturbations.
- Three row permutations and three expert permutations, producing nine combined cases per wrapper.
- An independent FP64 reference:

```python
reference = hidden_states[row_perm].double() @ weight[expert_perm].double().T
```

The exact permutation check compared:

```python
gate(hidden_states[row_perm], weight[expert_perm])
```

against:

```python
gate(hidden_states, weight)[row_perm][:, expert_perm]
```

The probe used the actual gate classes for Bailing MoE, Bailing MoE Linear, Ernie4, LLaDA2, and MiMo-V2. Kimi Linear’s model gate is `ReplicatedLinear`, so that layer was tested directly.

## Results

| Wrapper | Input dtype | Output dtype | Cases | Exact permutation | Max abs vs FP64 |
|---|---:|---:|---:|---:|---:|
| Bailing MoE | BF16 | BF16 | 9 | yes | 0.02935791015625 |
| Bailing MoE Linear | BF16 | BF16 | 9 | yes | 0.02935791015625 |
| LLaDA2 | BF16 | BF16 | 9 | yes | 0.02935791015625 |
| Ernie4 | FP32 | FP32 | 9 | yes | 4.76837158203125e-07 |
| MiMo-V2 | BF16 | FP32 | 9 | yes | 0.0 |
| Kimi Linear | FP32 | FP32 | 9 | yes | 4.76837158203125e-07 |

All 54 combined row/expert permutation cases passed the exact permutation check.

## Unsupported boundary

Ernie4’s current gate creates an FP32 weight when the default dtype is FP32, but does not cast BF16 hidden states. Passing BF16 input therefore fails before the GEMM with:

```text
expected mat1 and mat2 to have the same dtype, but got: BFloat16 != Float
```

That boundary was not treated as a permutation mismatch. Ernie4 was tested with its supported FP32 input boundary.

## Reproduction

Run the probe with:

```bash
PYTHONPATH=/job/sglang/python /opt/venv/bin/python - <<'PY'
from types import SimpleNamespace
import torch
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.models.bailing_moe import BailingMoEGate
from sglang.srt.models.bailing_moe_linear import BailingMoEGate as BailingLinearMoEGate
from sglang.srt.models.ernie4 import MoEGate as Ernie4MoEGate
from sglang.srt.models.llada2 import LLaDA2MoeGate
from sglang.srt.models.mimo_v2 import MoEGate as MiMoV2MoEGate

rows, hidden, experts = 7, 16, 8
row_perms = [
    torch.tensor([5, 0, 6, 2, 4, 1, 3], device="cuda"),
    torch.tensor([0, 6, 5, 4, 3, 2, 1], device="cuda"),
    torch.tensor([6, 5, 4, 3, 2, 1, 0], device="cuda"),
]
expert_perms = [
    torch.tensor([6, 2, 7, 0, 5, 3, 1, 4], device="cuda"),
    torch.tensor([0, 7, 6, 5, 4, 3, 2, 1], device="cuda"),
    torch.tensor([7, 1, 3, 5, 0, 2, 4, 6], device="cuda"),
]
row_idx = torch.arange(rows, device="cuda").view(-1, 1)
col_idx = torch.arange(hidden, device="cuda").view(1, -1)
x32 = 1.0 + 0.125 * ((row_idx + col_idx) % 5 - 2) + 0.0009765625 * ((row_idx * 3 + col_idx) % 3)
xbf16 = x32.to(torch.bfloat16)
expert_idx = torch.arange(experts, device="cuda").view(-1, 1)
weight_col = torch.arange(hidden, device="cuda").view(1, -1)
weight = 0.5 + 0.03125 * ((expert_idx + weight_col) % 7 - 3) + 0.000244140625 * ((expert_idx * 5 + weight_col) % 2)

configs = {
    "bailing_moe": SimpleNamespace(num_experts=experts, hidden_size=hidden, moe_router_enable_expert_bias=False),
    "bailing_moe_linear": SimpleNamespace(num_experts=experts, hidden_size=hidden, moe_router_enable_expert_bias=False),
    "llada2": SimpleNamespace(num_experts=experts, hidden_size=hidden, moe_router_enable_expert_bias=False),
    "ernie4": SimpleNamespace(moe_num_experts=experts, hidden_size=hidden),
    "mimo_v2": SimpleNamespace(n_routed_experts=experts, hidden_size=hidden, topk_method="noaux_tc"),
}
gates = {
    "bailing_moe": BailingMoEGate(config=configs["bailing_moe"], params_dtype=torch.float32).to("cuda"),
    "bailing_moe_linear": BailingLinearMoEGate(config=configs["bailing_moe_linear"], params_dtype=torch.float32).to("cuda"),
    "llada2": LLaDA2MoeGate(config=configs["llada2"], params_dtype=torch.float32).to("cuda"),
    "ernie4": Ernie4MoEGate(config=configs["ernie4"]).to("cuda"),
    "mimo_v2": MiMoV2MoEGate(config=configs["mimo_v2"], quant_config=None).to("cuda"),
    "kimi_linear": ReplicatedLinear(hidden, experts, bias=False, quant_config=None, prefix="gate").to("cuda"),
}
input_dtype = {
    "bailing_moe": torch.bfloat16,
    "bailing_moe_linear": torch.bfloat16,
    "llada2": torch.bfloat16,
    "ernie4": torch.float32,
    "mimo_v2": torch.bfloat16,
    "kimi_linear": torch.float32,
}
for name, gate in gates.items():
    x = xbf16 if input_dtype[name] == torch.bfloat16 else x32
    for row_perm in row_perms:
        for expert_perm in expert_perms:
            with torch.no_grad():
                gate.weight.copy_(weight)
                base = gate(x)
                if isinstance(base, tuple):
                    base = base[0]
                gate.weight.copy_(weight[expert_perm])
                got = gate(x[row_perm])
                if isinstance(got, tuple):
                    got = got[0]
            expected = base[row_perm][:, expert_perm]
            assert torch.equal(got, expected), name
            reference = x[row_perm].double() @ weight[expert_perm].double().T
            assert torch.allclose(got.double(), reference, atol=0.03, rtol=0.03), name
```

## Conclusion

The row/expert permutation property is preserved by all supported model gate wrappers tested on gfx942. No production change is warranted.
