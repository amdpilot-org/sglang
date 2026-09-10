# ROCm router GEMM execution-representation validation

## Scope

This is a distinct follow-up to amdpilot-org/sglang issue 233. It does not repeat the router-precision work in mirror PR 332 or the permutation work in mirror PR 354. The uncovered case tested here is tensor execution representation: a supported row-sliced activation versus representations that the M=1 AITER skinny kernel cannot read correctly.

The relevant operation is `aiter_dsv3_router_gemm(hidden_states, weight)` from sgl-project/sglang issue 38695. The SM90+ `tiny_gemm_bf16` path is CUDA-only on this stack, so the installed ROCm AITER dispatcher is the relevant gfx942 control.

## Environment

- Image: `amdpilotv2/open-job-mi300:jit-config-readable-35122-260909`
- Local image ID: `sha256:dfc9419089c338b5712da4841768b38b1ab79f3da41f8c58c3cd4dfcc1147ff1`
- GPU: one AMD Instinct MI300X, `gfx942:sramecc+:xnack-`, 304 compute units
- Python: `/opt/venv/bin/python`
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`
- HIP: `7.2.26015-fc0010cf6a`
- Mirror base commit: `0084030179bfba86bfeb6d43f7997d4076329d2c`
- Router helper: `python/sglang/srt/layers/rocm_linear_utils.py`
- AITER dispatcher: `/sgl-workspace/aiter/aiter/tuned_gemm.py`
- Native modules: `/sgl-workspace/aiter/aiter/jit/module_aiter_core.so` and `/sgl-workspace/aiter/aiter/jit/module_custom.so`

## Installed-source baseline

The first GPU baseline used the preinstalled source at commit `8eaffdf382e06b7dc50fc5c76cc5aad9c36bb0e4`, not the mirror checkout. It called `tgemm.mm` with BF16 inputs and `otype=torch.float32` for `[M, 256] @ [256, 7168].T`, using an independent CPU FP64 reference and 16-element sentinel guards around each input.

| M | Max abs error vs FP64 | Max abs error vs BF16-rounded FP64 | AITER BF16→FP32 (µs) | FP32 upcast (µs) | Dispatch |
|---:|---:|---:|---:|---:|---|
| 1 | 0.495880 | 0.0 | 50.468 | 35.357 | skinny solution 2 |
| 8 | 0.608185 | 0.0 | 39.355 | 31.890 | torch solution 0 |
| 16 | 0.919769 | 0.0 | 56.923 | 46.283 | torch solution 0 |

Timing used CUDA events with three warmups and ten timed iterations. All sentinel guards remained unchanged.

The stock gfx942 fallback is an important caveat: despite requesting FP32 output, its values are exactly BF16-quantized. The installed-source baseline is recorded in `/job/baseline-first.json` and is not evidence for later checkout changes.

## Pre-fix representation evidence

Before the host-side checks, the mirror base commit was probed with the same independent CPU FP64 reference:

- Row-sliced activations with stride `(7176, 1)` matched the reference at M=1, 8, and 16 and left sentinel guards unchanged.
- An inner-strided activation with stride `(14336, 2)` produced max abs error `424.0` at M=1. The M=1 skinny solution ignored the inner stride. M=8 used the Torch solution and matched.
- A row-sliced weight with stride `(7176, 1)` produced max abs error `369.5` at M=1. M=8 used the Torch solution and matched.
- An inner-strided weight with stride `(14336, 2)` produced max abs error `372.0` at M=1. M=8 used the Torch solution and matched.

Raw pre-fix values and commands are in `pre-fix-evidence.json`.

## Change

`aiter_dsv3_router_gemm` now rejects an activation whose inner stride is not one and a non-contiguous weight with clear `RuntimeError` messages. Row-sliced activations remain supported because their inner stride is one and the M=8/M=16 Torch dispatch handles the wider row stride.

The change is host-side validation only. It does not add a copy, alter AITER dispatch, introduce input/output aliasing, change output allocation or static graph addresses, or change the `otype=hidden_states.dtype` contract.

## Candidate results

The candidate harness uses BF16 `[M, 7168]` hidden states, BF16 `[256, 7168]` weights, an independent CPU FP64 reference rounded to BF16, and sentinel guards around both inputs.

| Representation | M | Stride | Max abs error | Mean time (µs) | Dispatch |
|---|---:|---|---:|---:|---|
| contiguous | 1 | `(7168, 1)` | 0.0 | 46.940 | skinny 2 |
| contiguous | 8 | `(7168, 1)` | 0.0 | 35.446 | torch 0 |
| contiguous | 16 | `(7168, 1)` | 0.0 | 49.819 | torch 0 |
| row-sliced | 1 | `(7176, 1)` | 0.0 | 70.643 | skinny 2 |
| row-sliced | 8 | `(7176, 1)` | 0.0 | 49.891 | torch 0 |
| row-sliced | 16 | `(7176, 1)` | 0.0 | 55.215 | torch 0 |

The new representation gate uses `atol=1.0`, `rtol=1e-2`, with BF16 output; no existing numerical gate was changed. All sentinel guards passed. The unsupported variants now fail clearly:

- Inner-strided activation: `aiter_dsv3_router_gemm requires hidden_states to have a unit inner stride`
- Non-contiguous weight: `aiter_dsv3_router_gemm requires a contiguous weight`

Raw results are in `execution-representation-results.json`.

## Reproduction

```bash
PYTHONPATH=python /opt/venv/bin/python -m pytest -q \
  test/registered/amd/test_rocm_router_gemm_execution_representations.py

PYTHONPATH=python /opt/venv/bin/python \
  reports/j-00f7a4657fb9/validate_execution_representations.py
```

The focused GPU test result was `3 passed, 3 subtests passed`. The validation harness also passed and wrote its JSON result.

## Not done

- No full model weights were downloaded.
- No AITER tuning job or toolchain replacement was run.
- No unbounded stress or GPU burn was used.
- Packed `is_shuffled` weights were not forced through this helper because it does not document or construct that representation; the supported non-contiguous activation case was the uncovered scope.
- No broad end-to-end model accuracy run was performed.
