# MI300X sparse MLA grid-boundary chunking report

## Scope

This report is a separate follow-up to amdpilot-org/sglang issue 226. It does not
repeat the already-fulfilled 65,535-row admission guard and full-versus-split
coverage in amdpilot-org/sglang PR 326.

The tested property is whether the supported Aiter sparse MLA decode path on one
MI300X (`gfx942`) preserves its operation contract when a legal launch at the
65,535-row grid boundary is decomposed into chunks, and whether a 65,536-row
single launch is supported.

Reference context:

- Upstream issue: sgl-project/sglang issue 34947
- Upstream guard change: sgl-project/sglang PR 34948
- Prior mirror issue: amdpilot-org/sglang issue 226
- Prior fulfilled mirror change: amdpilot-org/sglang PR 326
- Current mirror issue: amdpilot-org/sglang issue 276

## Environment

- Image: `amdpilotv2/open-job-mi300:jit-config-readable-260909-banff5`
- Operator-provided local image ID:
  `sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1`
- GPU: one AMD Instinct MI300X, compute capability `9.4` (`gfx942`)
- Python: `/opt/venv/bin/python`
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`
- HIP: `7.2.26015-fc0010cf6a`
- Torch path: `/opt/venv/lib/python3.10/site-packages/torch/__init__.py`
- Triton path: `/opt/venv/lib/python3.10/site-packages/triton/__init__.py`
- Aiter path: `/sgl-workspace/aiter/aiter/__init__.py`
- Persistent checkout: `/job/sglang`
- Persistent checkout base: `0084030179bfba86bfeb6d43f7997d4076329d2c`

## Installed-source baseline

The first GPU execution used the preinstalled source, not the persistent
checkout, and is therefore not proof for later checkout changes.

- Installed SGLang commit:
  `8eaffdf382e06b7dc50fc5c76cc5aad9c36bb0e4`
- Installed test:
  `/sgl-workspace/sglang/test/registered/kernels/ops/attention/test_flash_mla_backends.py`
- Direct control:
  `TestSparseDecodeTritonVsTorch._run`
- Comparison: independent pure-PyTorch sparse decode reference
- Numerical gate: `atol=0.05`, `rtol=0.05`
- Result: passed
- Maximum absolute difference: `1.8367099231598242e-40`
- First GPU execution elapsed time: `10.025602887850255 s`
- Timing method: `time.perf_counter` around one cold direct kernel execution
  with `torch.cuda.synchronize` before and after

The stock pytest form of that test is skipped on `gfx942` because its class gate
requires SM120. The direct control bypasses only that architecture gate and runs
the same kernel and reference.

## Persistent-checkout matrix

The persistent-checkout harness is
`reports/j-87321e5569e5/boundary_matrix.py`. It preserves the stock Aiter
sparse-MLA operation contract:

- Aiter kernel: `aiter.mla.mla_decode_fwd`
- Heads: `16`
- Head dimensions: query/key `576`, value `512`
- Page size: `1`
- KV length: `2048`
- Top-k values: `128` and `256`
- Original per-request `kv_indptr`
- Persistent page table
- Stock Triton request-index-to-global-index conversion
- Bfloat16 query/KV and output
- Independent FP32 gathered-key softmax and value reduction
- Numerical gate: `atol=1.6e-1`, `rtol=1.6e-1`
- Sentinel: NaN-filled output buffers, checked for complete writes
- Timing: `time.perf_counter` around each launch with
  `torch.cuda.synchronize` before and after; no warmup or repeated work

Finite adversarial matrix:

- Rows: `4`, `65535`, and `65536`
- Top-k: `128` and `256`
- Full single-launch control
- `[32768, remainder]`
- `[1, remainder]`
- Additional above-boundary splits for 65,536 rows: `[65535, 1]`

Raw results are in `reports/j-87321e5569e5/boundary-results.json`.

## Result

All six full launches and all split launches passed the unchanged numerical
gate. Every output was finite and every NaN sentinel was overwritten.

- Maximum reference absolute difference: `0.0035660862922668457`
- Full-launch errors: none
- Split-launch errors: none
- 65,535-row full launch: supported
- 65,536-row full launch: supported
- `[32768, remainder]` splits: supported
- `[1, remainder]` splits: supported

Large chunk splits were bitwise identical to their corresponding full-launch
rows. One-row splits at 65,535 and 65,536 rows were not always bitwise
identical, but all remained well inside the numerical gate:

- Maximum full-versus-split absolute difference: `0.00390625`
- Mean full-versus-split absolute difference range:
  approximately `0.00024` to `0.00033`
- Different elements per one-row output: `4748` to `4897` of `8192`

Those differences are consistent with bfloat16 ULP-sized variation from a
different reduction or split schedule for a one-row launch. They do not
demonstrate a violation of the operation contract or the preserved numerical
gate, so no production code change is made.

## Unsupported and neighboring controls

The following boundaries are recorded honestly rather than forced:

- `import flashinfer` fails with `ModuleNotFoundError`.
- `from sgl_kernel import flash_mla_ops` fails with `ImportError`.
- Aiter sparse MLA with `heads=4` fails its supported-kernel assertion on
  `gfx942`.
- The CUDA-only `trtllm-gen` path is not claimed or tested as a ROCm kernel.

The supported neighboring control is the stock Aiter sparse MLA test with 16
heads, top-k 128, a 2,048-token pool, page size 1, and the stock index
conversion path. That control passed before the boundary matrix was run.

## Reproduction

Installed-source baseline:

```bash
/opt/venv/bin/python /job/baseline-first.py
```

Persistent-checkout boundary matrix:

```bash
cd /job/sglang
/opt/venv/bin/python reports/j-87321e5569e5/boundary_matrix.py \
  --output reports/j-87321e5569e5/boundary-results.json
```

No full model weights were downloaded, no environment replacement was performed,
and no artificial GPU burn, unbounded loop, sleep loop, or repeated work was
used.
