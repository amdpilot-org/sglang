# gfx942 c128 cold/warm compiled-path reuse

## Scope

This is a bounded, single-GPU follow-up to amdpilot-org/sglang issue 222 and PR 251. PR 251 covered one fixed ragged planner shape with graph replay and stream ordering. This investigation extends the uncovered case to the real c128 prefill compressor across a finite shape sequence, checking cold versus warm JIT compiled-path reuse, FP32 and BF16 dtype contracts, non-aliasing storage, static graph addresses, and sentinel-protected output and state storage.

The tested mirror commit is `0084030179bfba86bfeb6d43f7997d4076329d2c`. It already contains the final upstream fix from sgl-project/sglang PR 32467, so this work does not duplicate or modify that kernel fix.

## Environment

- GPU: one AMD Instinct MI300X, `gfx942`, driver `6.19.14.31400000`
- Python: `/opt/venv/bin/python`
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`
- ROCm/HIP: `7.2.26015-fc0010cf6a`
- Assigned image: `amdpilotv2/open-job-mi300:jit-config-readable-260909-banff5`
- Assigned local image ID: `sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1`
- Image verification: no Docker socket or image metadata API is exposed in the container; the assigned ID is recorded rather than inferred from the hostname.
- Source: `/job/sglang/python/sglang/kernels/jit/csrc/deepseek_v4/c128_v2.cuh`
- Planner source: `/job/sglang/python/sglang/kernels/jit/csrc/deepseek_v4/c_plan.cuh`
- Native modules and raw results: `reports/j-c2ac3031e855/results.json`

## Installed-Source Baseline

The installed source is a different revision and is not proof for the mirror checkout. Its direct c128 ragged GPU-input planner call failed with:

```text
torch.AcceleratorError: HIP error: an illegal memory access was encountered
```

Uniform c128, ragged c4, and legacy CPU-input/GPU-output planner controls also failed with illegal-address errors. The supported neighboring installed c4 numerical control `test_prefill_no_context[4-paged]` passed against its existing FP64 reference:

```text
1 passed, 2 warnings in 9.77s
```

The complete early evidence, commands, and first-call elapsed time are in `/job/baseline-first.json`.

## Experiment

The new fixture uses the real `compress_forward` c128 prefill path with:

- Shape sequence: `[(128,128)]`, `[(128,128),(128,64)]`, and `[(128,128),(128,64),(256,128)]`
- Query-token counts: 128, 192, and 320
- Valid compress plans: 1, 2, and 3
- Supported dtypes: FP32 and BF16
- Independent row-by-row PyTorch reference for compressed output and final state
- Sentinel guard rows around both output and state storage
- Distinct input, APE, state, output, and plan allocations
- One graph capture and four replays per shape and dtype
- Static-address checks before and after graph capture
- A clear rejection test for unsupported `torch.complex64`

The independent reference reconstructs each 128-row source window from the documented plan fields, applies the plan write rows, computes the softmax-weighted KV reduction in FP32, and casts to the output dtype. It does not call the native compressor.

## Results

All numerical, buffer, sentinel, non-alias, static-address, graph-replay, and unsupported-dtype checks passed.

| dtype | cold first call | warm median 128 | warm median 192 | warm median 320 | max output error |
|---|---:|---:|---:|---:|---:|
| FP32 | 4.006 s | 0.03713 ms | 0.03699 ms | 0.04739 ms | 7.15e-7 |
| BF16 | 3.738 s | 0.04694 ms | 0.05076 ms | 0.05055 ms | 0.0 |

The first call for each dtype includes JIT compilation. Later shapes reuse the same dtype-specialized native module without recompilation. The bounded matrix uses 10 synchronized wall-clock calls and one CUDA-event measurement per shape. Four graph replays per shape complete in approximately 0.09–0.18 ms total.

Profiler dispatch records the expected native kernels:

```text
sglang::flash_c128_prefill<512l, float, float, float, false>
sglang::write_c128_prefill<512l, float, float, float, false>
sglang::flash_c128_prefill<512l, __hip_bfloat16, __hip_bfloat16, __hip_bfloat16, false>
sglang::write_c128_prefill<512l, __hip_bfloat16, __hip_bfloat16, __hip_bfloat16, false>
```

## Reproduction

Run the focused test:

```bash
PYTHONPATH="$PWD/python" \
SGLANG_JIT_CACHE_DIR=/tmp/sglang-cache-j-c2ac3031e855 \
AMD_SERIALIZE_KERNEL=3 \
/opt/venv/bin/python -m pytest -q \
  test/registered/kernels/ops/attention/test_deepseek_v4_compress_cold_warm_reuse.py
```

The raw timing, profiler, source, and native-module paths are retained in `reports/j-c2ac3031e855/results.json`.

## Limitations

- This is operator-level evidence only; it makes no full-model, multi-GPU, TP/DP, or distributed-serving claim.
- Timing is a small bounded matrix, not an end-to-end throughput benchmark.
- Results are specific to one MI300X and the qualified Torch/ROCm stack.
- The installed-source planner failures are reported as environment evidence and are not attributed to the already-fixed mirror checkout.
