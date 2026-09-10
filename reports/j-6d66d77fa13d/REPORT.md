# Bounded BF16/FP8-KV reduced-block study on MI300X

## Result

This is a decode-sized dummy-model latency study, not a model-quality claim. The
supported Aiter control ran the real `ProjectedDenseAttention` block through
`RadixAttention` for four recurrent decode steps. All six workload cases passed
the fixed BF16 and FP8 numerical gates against independent dequantized
references. The DSA FlashMLA FP8 path was not runnable in this image, so it is
reported as unsupported rather than replaced by a speculative fix.

## Environment

- GPU: one AMD Instinct MI300X, `gfx942:sramecc+:xnack-`, UUID
  `34333965-3031-6163-3332-323164383838`, 304 CUs, 206,141,652,992 bytes HBM.
- Python: `/opt/venv/bin/python` (3.10.12).
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`; HIP
  `7.2.26015-fc0010cf6a`.
- Image reference supplied by the operator:
  `amdpilotv2/open-job-mi300:jit-config-readable-260909-banff5`, local image
  ID `sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1`.
  No Docker socket or image metadata was available inside the job, so this ID
  was not independently verified.
- Installed-source context: commit
  `8eaffdf382e06b7dc50fc5c76cc5aad9c36bb0e4`, Python package
  `/sgl-workspace/sglang/python/sglang`.
- Tested mirror checkout: commit
  `0084030179bfba86bfeb6d43f7997d4076329d2c`, Python package
  `/job/sglang/python/sglang`.
- Native/imported paths: Aiter
  `/sgl-workspace/aiter/aiter/jit/module_aiter_core.so`, Torch
  `/opt/venv/lib/python3.10/site-packages/torch/__init__.py`, and unavailable
  FlashMLA extension
  `/opt/venv/lib/python3.10/site-packages/sgl_kernel/flash_mla.py`.
- Final build caches were job-private and outside `/job`:
  `/tmp/sglang-cache-j-6d66d77fa13d/aiter` and
  `/tmp/sglang-cache-j-6d66d77fa13d/triton`.

## First installed-source GPU baseline

The first GPU execution used the preinstalled source and an existing ROCm
relevant Triton decode-attention test. It completed before cloning or editing:

```bash
timeout 300 /opt/venv/bin/python -m pytest -q -s -p no:cacheprovider \
  /sgl-workspace/sglang/test/registered/attention/test_triton_attention_kernels.py::TestTritonAttention::test_decode_attention \
  /sgl-workspace/sglang/test/registered/attention/test_triton_attention_kernels.py::TestTritonAttention::test_grouped_decode_attention
```

Result: `2 passed`; pytest reported 24.82 seconds and wall time was 28.427
seconds. The dense cases compare the Triton kernel with an independent float32
stable-softmax reference using `atol=1e-2, rtol=1e-2`. The grouped cases
compare normal and grouped Triton decode outputs using cosine similarity above
0.99 and AMD-CI `allclose(atol=5e-2)`. Full details are in
`/job/baseline-first.json`. This installed-source baseline is not evidence for
later checkout changes.

## Upstream context

- `sgl-project/sglang` issue 31783 is the open “Quantization 2026 H2” roadmap.
  Its capability-registry and active-blocker sections call for consistent
  backend/KV-dtype/platform validation.
- Upstream PR 32576 is open and proposes a declarative DSA backend compatibility
  table. It was not applied or tested because the required DSA FlashMLA kernel
  is unavailable in this image.
- Upstream PR 31346 is merged and fail-fasts CUDA tilelang DSA with FP8 KV. It
  was not duplicated.
- No upstream issue, PR, or comment was posted or modified.

## DSA support probes

The installed-source DSA FP8 tests were bounded to 300 seconds:

```bash
timeout 300 /opt/venv/bin/python -m pytest -q -s -p no:cacheprovider \
  /sgl-workspace/sglang/test/registered/attention/unittests/dsa/test_dsa.py \
  -k 'fp8_decode or fp8_prefill'
```

They failed during `DSATokenToKVPool` allocation because the HIP legacy path
requires page size 1 but the fixture uses page size 64:
`AssertionError: HIP legacy DSA path requires page_size == 1, got 64`.

A checkout probe then used page size 1 with `PYTHONPATH=/job/sglang/python` and
`dsa_decode_backend="flashmla_kv"`. It reached metadata setup but failed with:

```text
ImportError: cannot import name 'flashmla_ops' from 'sgl_kernel'
ImportError: Failed to load sgl_kernel.flashmla_ops extension.
```

The exact probe is recorded in `/job/dsa-page1-checkout-probe.log`. No DSA
output or latency is claimed.

## Supported dummy-model benchmark

### Dimensions and state

- Block: real `ProjectedDenseAttention` with Q/K/V/O projections and
  `RadixAttention`; Aiter `AiterAttnBackend`.
- Batch/prefix matrix, exactly six cases: `(1,128)`, `(1,512)`, `(2,128)`,
  `(2,512)`, `(4,128)`, `(4,512)`.
- Page size 16; 4 query heads; 2 KV heads; head dimension 128; hidden size 512.
- Four recurrent decode steps per sequence. Step `i` uses prefix length
  `base_prefix + i`, writes the current K/V slot, and attends over the actual
  cache state.
- Query and block output are BF16. BF16 KV uses `torch.bfloat16`. FP8 KV uses
  `torch.float8_e4m3fnuz` on this ROCm stack.
- Explicit FP8 scales: K/query `0.02`, V `0.03`.
- Locally generated block weights: 1,572,864 bytes. No model weights were
  downloaded.

### Independent references and gates

- BF16 reference: read the actual K/V cache, convert to float32, and compute
  stable softmax attention plus the output projection.
- FP8 reference: independently quantize the BF16 query with K scale using
  `scaled_fp8_quant`, dequantize it, dequantize actual cache bytes as
  `K * 0.02` and `V * 0.03`, and compute float32 attention plus output
  projection. The reference does not call the Aiter attention kernel.
- BF16 gate: all outputs finite, maximum absolute error at most 0.05, cosine at
  least 0.999.
- FP8 gate: all outputs finite, mismatch fraction at most 0.5% at
  `atol=0.15, rtol=0.15`, cosine at least 0.99.
- Correctness is checked after every recurrent step in the correctness pass.
  Warmup and measured sequences also compare all four step outputs with the
  stored references.

### Timing

- Three bounded warmup sequences and 15 measured sequences per case and dtype.
- Sixty real block-forward step samples and 15 full-sequence samples per
  case/dtype.
- Step latency uses CUDA events around `ProjectedDenseAttention.forward`;
  metadata construction and backend metadata initialization are excluded.
- Sequence latency uses CUDA events around all four steps and therefore
  includes metadata construction/init and Python dispatch between forwards.
- p95 uses linear interpolation. Raw step and sequence samples are retained in
  `results.json`.

### Observed results

```text
batch prefix kv   step_median_ms step_p95_ms max_abs_error min_cosine peak_allocated_bytes
1     128    BF16 0.324409       0.375271   0.000488      0.999994   84402176
1     128    FP8  0.391183       0.462049   0.002930      0.999667   84510720
1     512    BF16 0.487686       0.631746   0.000244      0.999994   91140608
1     512    FP8  0.646712       0.858849   0.001190      0.999681   91452416
2     128    BF16 0.368851       0.549212   0.000488      0.999994   87949312
2     128    FP8  0.463370       0.638111   0.003174      0.999696   88140288
2     512    BF16 0.654711       0.859283   0.000244      0.999995   99052032
2     512    FP8  0.889473       1.036837   0.001526      0.999708   99706368
4     128    BF16 0.494441       0.674997   0.000488      0.999994   93606912
4     128    FP8  0.560072       0.760151   0.003082      0.999617   93307904
4     512    BF16 0.606600       0.833277   0.000244      0.999995   115049984
4     512    FP8  0.710781       0.959681   0.001541      0.999699   113964544
```

All FP8 mismatch fractions were zero at the fixed FP8 tolerance. In this bounded
dummy-model study, FP8 step median and p95 were higher than BF16 for all six
cases. This is an observed latency result only.

### Allocation and dispatch

- The smallest cache uses pool size 176 slots and K/V buffer shape
  `[192, 2, 128]`; BF16 allocates 98,304 bytes per K or V buffer, while FP8
  allocates 49,152 bytes each.
- The largest cache uses pool size 2,144 slots and K/V buffer shape
  `[2160, 2, 128]`; BF16 allocates 1,105,920 bytes per K or V buffer, while
  FP8 allocates 552,960 bytes each.
- Every measured run stayed below 115,049,984 bytes of live allocation, well
  under the 48-GB limit.
- Dispatch counters observed four `unified_attention` calls per four-step
  correctness pass for every case/dtype. `paged_attention_ragged` was not used.
- BF16 used the standard KV write path. FP8 used
  `launch_reshape_and_cache_flash` four times per four-step pass (fused BF16 to
  FP8 cast/write).
- `kv_cache_is_vectorized_5d` was false and
  `use_triton_unified_attention` was true.

## Reproduction

Run from `/job/sglang`:

```bash
export PYTHONPATH=/job/sglang/python
export AITER_ROOT_DIR=/tmp/sglang-cache-j-6d66d77fa13d/aiter
export TRITON_CACHE_DIR=/tmp/sglang-cache-j-6d66d77fa13d/triton
export SGLANG_USE_AITER_UNIFIED_ATTN=1
export SGLANG_AITER_KV_CACHE_LAYOUT=nhd
timeout 1800 /opt/venv/bin/python \
  reports/j-6d66d77fa13d/benchmark_reduced_block.py \
  --output reports/j-6d66d77fa13d/results.json
```

The final run completed in 36.212 seconds and wrote the raw JSON artifact. The
DSA page-1 checkout probe completed in approximately 15.2 seconds and failed as
described above.

## Bounds and unfinished work

- Wall limit: 120 minutes; the bounded GPU work completed well inside it.
- Workload limit: six cases, each run in BF16 and FP8.
- Weight limit: locally generated weights were 1,572,864 bytes, below 4 GB.
- Live allocation limit: below 48 GB.
- No synthetic burn, unbounded loop, sleep loop, or full model weights.
- The DSA FlashMLA FP8 path remains untested because `flashmla_ops` is absent.
  A future image with that extension, or a supported HIP DSA backend, is needed.
- The sequence timing includes Python/metadata overhead and is not a standalone
  kernel timing. Step timing is the block-forward comparison.
- No production code was changed and no model-quality claim is made.
