# Reduced gfx942 block study

## Scope

This is a bounded, existing-configuration search on one AMD Instinct MI300X
(`gfx942`). It executes the real public SGLang operators as a reduced block:

1. `sglang.kernels.ops.layernorm.norm.rmsnorm`
2. `sglang.kernels.ops.activation._SILU_AND_MUL.forward`
3. `sglang.kernels.ops.attention.dsv4.linear_bf16_fp32`

The study compares each configuration against an independent Torch composition,
checks dtype and mutation contracts, and attributes per-operator versus
whole-chain cost. It does not modify production code.

Read-only context: upstream issue `sgl-project/sglang#29630` describes the
completed migration to the public `sglang.kernels.ops` surface used here. No
upstream issue, PR, or comment was posted or modified.

## Environment

- Image: `amdpilotv2/open-job-mi300:jit-config-readable-35122-260909`
- Local image ID: `sha256:dfc9419089c338b5712da4841768b38b1ab79f3da41f8c58c3cd4dfcc1147ff1`
- GPU: AMD Instinct MI300X, GUID `44877`, serial `692440004372`
- Python: `/opt/venv/bin/python`
- Torch: `/opt/venv/lib/python3.10/site-packages/torch/__init__.py`
- SGLang checkout: `/job/sglang`
- AITER: `/sgl-workspace/aiter/aiter`
- `sgl_kernel`: `/opt/venv/lib/python3.10/site-packages/sgl_kernel`
- Mirror base commit: `0084030179bfba86bfeb6d43f7997d4076329d2c`

## Workload

The block uses:

- Input and RMSNorm output: `[M, 8192]`, `torch.bfloat16`
- Activation input: `[M, 8192]`, `torch.bfloat16`
- Activation output: `[M, 4096]`, `torch.bfloat16`
- Projection weight: `[4096, 4096]`, `torch.bfloat16`
- Projection output: `[M, 4096]`, `torch.float32`

Finite representative cases:

| Case | `M` |
|---|---:|
| decode | 1 |
| small batch | 64 |
| prefill | 1024 |
| large prefill | 4096 |

Weight allocation is `33,570,816` bytes. Peak live allocation in the harness is
`201,342,976` bytes. Both remain below the requested 4 GiB and 48 GiB limits.

## Configurations

At most four supported existing configurations were considered. Three were
measured:

1. Direct SGLang JIT RMSNorm + AOT `silu_and_mul` + public bf16→fp32 projection
2. Direct SGLang JIT RMSNorm + AITER `silu_and_mul` + public bf16→fp32 projection
3. Direct SGLang JIT RMSNorm + Torch `silu_and_mul` + public bf16→fp32 projection

The public fused-op AITER RMSNorm configuration was rejected before measurement
because its current wrapper passes arguments in an incompatible order to
`aiter::rmsnorm2d_fwd`. The exact failure is preserved in
`aiter-norm-failure.log`.

## Accuracy and contracts

The independent Torch reference is:

```python
norm = input.float()
norm = norm * torch.rsqrt(norm.pow(2).mean(-1, keepdim=True) + eps)
norm = (norm * weight.float()).to(torch.bfloat16)
activation = F.silu(norm[..., :4096].float()).to(torch.bfloat16) * norm[..., 4096:]
projection = torch.mm(activation.float(), projection_weight.float().t())
```

Final gates, unchanged across configurations:

- RMSNorm: `atol=5e-2`, `rtol=2e-2`
- Activation: `atol=5e-2`, `rtol=2e-2`
- Projection: `atol=1e-1`, `rtol=8e-2`
- Whole chain: `atol=1e-1`, `rtol=8e-2`

An initial stricter gate (`atol=1e-2`, `rtol=1e-2` for norm and activation)
rejected large-case results because ordinary bf16 rounding exceeded it. That
run is preserved in `results-strict-gate.json`; it is not silently discarded.

Contracts checked for every measured case and configuration:

- Block input, RMSNorm weight, and projection weight remain unchanged.
- Explicit `out` identity is preserved for RMSNorm and activation.
- RMSNorm and activation outputs remain `torch.bfloat16`.
- Projection and whole-chain outputs remain `torch.float32`.

The AOT and Torch activation configurations pass every gate and contract. The
AITER activation configuration passes each operator locally, but its small
activation differences propagate through the projection and fail the whole-chain
gate. Its timings are reported for context but are not accepted as a passing
configuration.

## Timing method

Each target uses 10 warmup iterations followed by 30 measured iterations. Each
measurement is repeated five times. The reported value is the median across the
five repeats; uncertainty is reported as the interquartile range (IQR). Timing
uses CUDA events on the default stream and is therefore subject to shared
MI300X hardware and clock variability.

## Results

Median whole-chain times in milliseconds:

| Case | AOT activation | Torch activation | AITER activation |
|---|---:|---:|---:|
| decode | 0.058118 | 0.042624 | 0.047419 |
| small batch | 0.029620 | 0.042800 | 0.047050 |
| prefill | 0.099382 | 0.107648 | 0.098186 |
| large prefill | 0.293892 | 0.341726 | 0.290699 |

Whole-chain IQR range across all reported measurements is approximately
`0.00060–0.00820 ms`.

For the passing AOT configuration, projection accounts for:

- 58.3% of chain time at `M=1`
- 57.6% at `M=64`
- 77.3% at `M=1024`
- 77.9% at `M=4096`

For the passing Torch configuration, projection accounts for:

- 39.2% at `M=1`
- 39.4% at `M=64`
- 72.1% at `M=1024`
- 67.1% at `M=4096`

Projection is therefore the dominant large-batch bottleneck. At decode, the AOT
configuration is projection-bound, while the Torch configuration is slightly
activation-bound.

## Reproduction

Installed-source baseline:

```bash
/opt/venv/bin/python -m pytest -q \
  '/sgl-workspace/sglang/test/registered/kernels/ops/layernorm/test_rmsnorm.py::test_rmsnorm[True-dtype0-1-4096]'
```

The successful installed-source baseline passed in 30.813 seconds and is recorded
in `/job/baseline-first.json`.

Reduced-block benchmark:

```bash
cd /job/sglang
PYTHONPATH=/job/sglang/python /opt/venv/bin/python \
  reports/j-faa676883248/benchmark_block.py \
  --output reports/j-faa676883248/results.json
```

AITER RMSNorm failure control:

```bash
cd /job/sglang
PYTHONPATH=/job/sglang/python /opt/venv/bin/python - <<'PY'
import torch
from sglang.kernels.ops.layernorm import _RMSNORM
from sglang.kernels.spec import KernelBackend
x = torch.randn(4, 8192, device="cuda", dtype=torch.bfloat16)
w = torch.randn(8192, device="cuda", dtype=torch.bfloat16)
out = torch.empty_like(x)
_RMSNORM.forward(x, w, 1e-6, out, backend=KernelBackend.AITER)
PY
```

## Raw artifacts

- `results.json`: final gates, contracts, timings, and uncertainty
- `results-strict-gate.json`: initial stricter-gate run
- `aiter-norm-failure.log`: exact unsupported AITER RMSNorm error
- `benchmark_block.py`: bounded reproduction harness
- `baseline-first.json`: installed-source baseline metadata

The installed-source baseline is also saved outside the repository at
`/job/baseline-first.json`.
