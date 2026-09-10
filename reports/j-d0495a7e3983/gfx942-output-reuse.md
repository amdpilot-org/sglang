# gfx942 FP8 paged MQA logits output-reuse evidence

## Scope

- Upstream context: `sgl-project/sglang` issue 34718, read only.
- Follow-up context: `amdpilot-org/sglang` issue 225.
- Prior mirror PR 337 was tested at commit `8f7a8c01f3d70ebe6037dae00494a529560eccbe`; its two existing numerical tests passed.
- Prior mirror PR 356 was consulted for page-permutation coverage. This change does not repeat that work.
- The uncovered case here is fresh output storage versus safe reuse of one explicit output buffer across two distinct input batches.
- No upstream issue, PR, or comment was posted or modified.

## Environment

- Image: `amdpilotv2/open-job-mi300:jit-config-readable-35122-260909`
- Local image ID: `sha256:dfc9419089c338b5712da4841768b38b1ab79f3da41f8c58c3cd4dfcc1147ff1`
- GPU: one assigned AMD Instinct MI300X, `gfx942`, device capability `(9, 4)`.
- Python: `/opt/venv/bin/python` 3.10.12.
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`, HIP `7.2.26015-fc0010cf6a`.
- Triton: `3.7.0`.
- Installed SGLang import: `/sgl-workspace/sglang/python/sglang/__init__.py`.
- Installed AITER import: `/sgl-workspace/aiter/aiter/__init__.py`.
- Native AITER module: `/sgl-workspace/aiter/aiter/jit/module_aiter_core.so`.
- Persistent mirror base: `amdpilot-org/sglang` commit `0084030179bfba86bfeb6d43f7997d4076329d2c`.
- Build caches were kept under `/tmp/sglang-cache-j-d0495a7e3983`, outside `/job`.

## Installed-source baseline

The first GPU execution completed in `1.3237035693600774 s` wall time. The installed baseline used AITER's:

```text
aiter.ops.triton.attention.pa_mqa_logits.deepgemm_fp8_paged_mqa_logits
```

with `Preshuffle=False`, `KVBlockSize=1`, `ChunkK=256`, `next_n=1`, 32 heads, head dimension 128, batch size 2, and context lengths `[130, 193]`.

The independent reference dequantized the actual FP8 query and keys, applied the UE8M0 key scales, computed per-head dot products, applied ReLU and head weights, and summed heads. The unchanged numerical gate was `atol=2e-2, rtol=2e-2`.

| output path | maximum absolute error | mean absolute error |
|---|---:|---:|
| fresh output | `1.52587890625e-05` | `2.1258374545141123e-06` |
| reused output, first batch snapshot | `1.52587890625e-05` | `2.1258374545141123e-06` |
| reused output, second batch | `1.52587890625e-05` | `2.2631311367149465e-06` |

The installed baseline used three CUDA events per case after one warmup dispatch:

| case | timings (ms) |
|---|---|
| fresh output | `[0.2090820074081421, 0.08840300142765045, 0.07797999680042267]` |
| reused output | `[0.0734890028834343, 0.06719499826431274, 0.065591000020504]` |

The 16-element float32 sentinels before and after output storage remained unchanged. This installed-source baseline is recorded in `/job/baseline-first.json` and is not proof for later checkout changes.

## Checkout result

The added test uses the same native operation with `Preshuffle=False`, `KVBlockSize=1`, `ChunkK=128`, `next_n=1`, 32 heads, head dimension 128, batch size 2, output width 256, and context lengths `[130, 193]`.

It checks:

- fresh output and reused output against independent dequantized references;
- two distinct input batches through the same output buffer;
- bitwise equality of fresh and reused first-batch logits;
- unchanged output `data_ptr()` across both reused dispatches;
- float32 dtype, expected shape, contiguous layout, and CUDA device;
- non-overlap of output storage with query, fused KV, weights, and page-map storage;
- unchanged 16-element sentinels before and after output storage;
- clear `ValueError` rejection for float16 output and non-contiguous output.

All maximum absolute errors were `1.52587890625e-05`. Fresh and reused first-batch logits were bitwise equal. The reused address was stable and all guard sentinels remained unchanged.

The checkout timing matrix used three CUDA events per case after one warmup dispatch:

| case | timings (ms) |
|---|---|
| fresh output | `[0.09401600062847137, 0.07942300289869308, 0.07497300207614899]` |
| reused output | `[0.07974400371313095, 0.07397100329399109, 0.0751739963889122]` |

## Commands

```bash
cd /job/sglang
export PYTHONPATH=/job/sglang/python
export TRITON_CACHE_DIR=/tmp/sglang-cache-j-d0495a7e3983/triton
export TVM_FFI_CACHE_DIR=/tmp/sglang-cache-j-d0495a7e3983/tvm-ffi
export SGLANG_USE_AITER=1

/opt/venv/bin/python -m pytest \
  test/registered/amd/test_aiter_paged_mqa_logits_reuse.py \
  -q --tb=short -rA
```

Observed:

```text
2 passed, 3 warnings in 21.59s
```

Prior PR 337 candidate command, run from its detached worktree at commit `8f7a8c01f3d70ebe6037dae00494a529560eccbe`:

```bash
cd /tmp/sglang-pr337-j-d0495a7e3983
export PYTHONPATH=/tmp/sglang-pr337-j-d0495a7e3983/python
export TRITON_CACHE_DIR=/tmp/sglang-cache-j-d0495a7e3983/triton
export TVM_FFI_CACHE_DIR=/tmp/sglang-cache-j-d0495a7e3983/tvm-ffi
export SGLANG_USE_AITER=1

/opt/venv/bin/python -m pytest \
  test/registered/amd/test_aiter_paged_mqa_logits.py \
  -q --tb=short -rA
```

Observed:

```text
2 passed, 3 warnings in 20.22s
```

## Unsupported and negative evidence

- Direct DeepGEMM `fp8_paged_mqa_logits` is not installed in this ROCm image (`ModuleNotFoundError: No module named 'deep_gemm'`), and the upstream SM90/SM100 path is architecture-gated.
- An exploratory non-preshuffle `KVBlockSize=64`, 32-head variant produced `HIP_ERROR_ILLEGAL_ADDRESS` during synchronized execution. It was not forced through and is not claimed as a supported result.
- The test dispatch rejects unsupported float16 and non-contiguous output variants before native dispatch. It does not reinterpret or force an invalid output ABI.
- Production preshuffle and 64-token-block layouts remain unproven here.
- Only `next_n=1` is covered; PR 337 already covers `next_n=2` numerical behavior.
- The test checks the stable-address precondition for graph-style reuse by comparing `data_ptr()` across dispatches; it does not capture or replay a full CUDA/HIP graph.

## Limitations

- This is bounded single-GPU evidence, not a full-model or production eviction workload.
- No full model weights were downloaded, no toolchain was replaced, and no node-wide state was modified.
- No production kernel change is claimed. The supported reuse case already preserves the documented output contracts; this change adds regression coverage and evidence.
