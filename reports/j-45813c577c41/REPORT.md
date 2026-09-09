# ROCm static-FP8 RMSNorm producer-fusion investigation

Campaign: `repo-e2e-20260909`
Task: `j-45813c577c41`
Coordination tracker: `amdpilot-org/amdpilotv2` issue `402`
Result marker: `OPEN_TASK_REPORTED`

## Executive summary

The current ROCm/gfx942 path does **not** dispatch the existing static per-tensor FP8 RMSNorm producer fusion. `RMSNorm.forward_with_per_tensor_quant_fusion` exists in current `main`, but `python/sglang/srt/layers/layernorm.py` imports FlashInfer only for CUDA, XPU, or MUSA. On this HIP stack `_flashinfer_rmsnorm_quant_available` is `False`, and FlashInfer is not installed. A production `RMSNorm.forward(..., quant_linear=...)` call with a static per-tensor scale therefore falls back to unquantized BF16 output rather than returning the pre-quantized FP8 tuple.

The active ROCm producer fusion is `_fused_rmsnorm_fp8_per_token_quant` in `python/sglang/srt/layers/communicator.py`, backed by aiter `add_rmsnorm_quant`. It emits one dynamic FP8 scale per token, not the fixed static per-tensor scale required by the roadmap slice. Its output is not equivalent to separate RMSNorm followed by `static_quant_fp8`: exact FP8-byte equality and exact returned-scale equality fail at both tested shapes.

No already-working ROCm static per-tensor producer fusion was found, so no fix is claimed.

## Environment and paths

- Image: `amdpilotv2/open-job-mi300:jit-config-readable-260909-banff5`, local image ID `sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1`.
- GPU: one `AMD Instinct MI300X`, capability `(9, 4)` (`gfx942`), device count 1.
- Python: `/opt/venv/bin/python`, Python `3.10.12`.
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`; HIP `7.2.26015-fc0010cf6a`.
- Working source: `/job/sglang`, current-source commit `ffe98a4279ba6e42d1f87dc4eeb6edb4887b9ea4`.
- Source module: `/job/sglang/python/sglang/srt/layers/layernorm.py`.
- ROCm producer source: `/job/sglang/python/sglang/srt/layers/communicator.py`.
- aiter Python module: `/sgl-workspace/aiter/aiter/__init__.py`.
- aiter native RMSNorm-quant module: `/sgl-workspace/aiter/aiter/jit/module_rmsnorm_quant.so`.
- `sgl_kernel` module: `/opt/venv/lib/python3.10/site-packages/sgl_kernel/__init__.py`.
- Torch module: `/opt/venv/lib/python3.10/site-packages/torch/__init__.py`.

## Method

The synthetic validation uses BF16 RMSNorm inputs, BF16 residual inputs, BF16 norm weights, and no model weights. The fixed static reciprocal FP8 scale is `0.05`. Tests use `torch.float8_e4m3fnuz`, the ROCm FP8 dtype selected by the source wrapper.

Paths compared:

1. **Actual current ROCm fused producer:** `_fused_rmsnorm_fp8_per_token_quant(hidden_states, weight, epsilon, residual=residual)`, which returns FP8 plus one dynamic scale per token.
2. **Separate producer then static FP8:** `RMSNorm.forward(hidden_states, residual)` followed by `static_quant_fp8(normed, static_scale)`.

Shapes are `(128, 4096)` and `(8192, 4096)`. Timing uses 25 warmup calls, then five repeats of 200 calls bounded by `torch.cuda.Event`. The reported per-call time includes wrapper allocations and launch overhead; it is warm end-to-end operator timing, not an isolated pure-kernel timing.

### Fixed numerical gates

These gates were fixed before measurement and were not changed:

- Exact FP8 byte equality: `100.0%`.
- Maximum dequantized absolute error: `<= 0.10`.
- Dequantized cosine similarity: `>= 0.999`.
- Returned scale must exactly equal the fixed static per-tensor scale.

## Current-source raw results

| Shape | Fused per-token median | Separate + static median | Separate minus fused | Exact FP8 bytes equal | Max dequant error | Mean dequant error | Relative L2 | Cosine | Fused scale min/max/mean | Static scale |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `128 x 4096` | `35.270276 us` | `77.701297 us` | `42.431021 us` | `479 / 524288` (`0.091362%`) | `0.356094` | `0.024715` | `0.037863` | `0.999292` | `0.013906 / 0.022226 / 0.016660` | `0.05` |
| `8192 x 4096` | `60.242572 us` | `90.323858 us` | `30.081286 us` | `29900 / 33554432` (`0.089109%`) | `0.378151` | `0.024477` | `0.037665` | `0.999299` | `0.013244 / 0.025418 / 0.016687` | `0.05` |

Gate outcomes at both shapes:

- Exact FP8 byte equality: **fail**.
- Maximum dequantized absolute error: **fail**.
- Dequantized cosine similarity: **pass**.
- Exact static returned-scale equality: **fail**.

The per-token fused path is faster than the separate producer plus static quantization in this synthetic warm timing, but it is not the requested static-FP8 semantic path and does not satisfy the fixed equivalence gates.

## Dispatch observations

- `_flashinfer_rmsnorm_quant_available`: `False`.
- Production `RMSNorm.forward` with a static per-tensor consumer scale: returns BF16, not a pre-quantized tuple; `fell_back_to_unquantized_bf16=true`.
- Direct `RMSNorm.forward_with_per_tensor_quant_fusion` call: raises `NameError`, `_flashinfer_fused_add_rmsnorm_quant` is not defined.
- The installed aiter `rmsnorm_quant`/`add_rmsnorm_quant` API emits dynamic scales; it does not accept a fixed static per-tensor scale as input.

## Candidate revision

A bounded one-line candidate at commit `571950e02f7da8d552b1de9bd1d459d74343528c` changed the FlashInfer import gate to include HIP. It did **not** enable static fusion on this stack:

- `_flashinfer_rmsnorm_quant_available`: `False`.
- Production dispatch still fell back to unquantized BF16.
- Direct static fusion still raised the same `NameError`.

The candidate is not a fix because FlashInfer is absent from the qualified ROCm/Torch stack. Candidate timing is not compared against current-source timing because the separate candidate run showed run-to-run variance; only its dispatch result is claimed.

## Read-only issue and related-change context

Issue `sgl-project/sglang` `31504` was read without modification. Its current description and comments identify the static-FP8 producer-fusion roadmap and note that the existing ROCm norm fusion is per-token. Related reviewed changes were:

- PR `36501`: CUDA SM90/SM100 FlashInfer allreduce static-FP8 fusion; not a gfx942 ROCm producer path.
- PR `36805`: current `main` source containing the static per-tensor RMSNorm fusion method, gated to CUDA/XPU/MUSA through FlashInfer.
- PR `32443`: CUDA gated RMSNorm plus group-128 dynamic FP8 quantization.
- PR `34502`: ROCm gfx95 per-channel dynamic FP8 RMSNorm fusion.
- PR `28932`: ROCm aiter SiluAndMul plus dynamic per-token FP8 quantization.

No reviewed candidate supplied an already-working ROCm static per-tensor RMSNorm or activation producer fusion, so no working fix was duplicated.

## Commands

The main commands were:

```bash
git clone --depth=200 https://github.com/amdpilot-org/sglang.git /job/sglang
gh issue view 31504 --repo sgl-project/sglang --json number,title,body,comments,url,state,author,createdAt,updatedAt,labels
PYTHONPATH=/job/sglang/python SGLANG_USE_AITER=1 \
  /opt/venv/bin/python reports/j-45813c577c41/benchmark_rocm_static_fp8_rmsnorm.py \
  --output reports/j-45813c577c41/results.json
```

The candidate gate change was committed on a local candidate branch, measured, and then the working tree was returned to `main`; the candidate was not pushed or delivered as a fix.

## Reproduction

From `/job/sglang`:

```bash
PYTHONPATH=/job/sglang/python SGLANG_USE_AITER=1 \
  /opt/venv/bin/python reports/j-45813c577c41/benchmark_rocm_static_fp8_rmsnorm.py \
  --output reports/j-45813c577c41/results.json
```

The raw JSON is retained in `reports/j-45813c577c41/results.json`.

## Uncertainty and undone work

- No new ROCm static per-tensor fused kernel was implemented, and FlashInfer was not installed; installing another stack would violate the qualified-image constraint.
- Timing includes wrapper allocations and launch overhead. A CUDA/HIP graph or preallocated-output microbenchmark could isolate pure kernel time, but would not change the dispatch or semantic-scale result.
- No full model or model weights were downloaded or used.
