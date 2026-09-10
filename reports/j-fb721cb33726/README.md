# Dense FP8 gfx942 lifecycle evidence

## Scope

This records a dense block-FP8 linear weight lifecycle check on one assigned AMD
Instinct MI300X (`gfx942`). It intentionally does not claim coverage for AWQ or
GPTQ packing, INT8, MoE dispatch, MTP scheme selection, or MXFP8 dense linear.

Upstream issue `sgl-project/sglang` issue number `15194` is the broad
quantization roadmap. Its current description and comments discuss scheme
structure, backend/kernel organization, and related format work; they do not
provide a specific dense block-FP8 scale fix to duplicate. No upstream issue,
PR, or comment was posted or changed.

## Environment

- Image: `amdpilotv2/open-job-mi300:jit-config-readable-260909-banff5`, local
  image ID `sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1`.
- GPU: one `AMD Instinct MI300X`, CUDA/HIP capability `(9, 4)`, `gfx942`.
- Interpreter: `/opt/venv/bin/python`.
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`, HIP
  `7.2.26015-fc0010cf6a`, module
  `/opt/venv/lib/python3.10/site-packages/torch/__init__.py`.
- Installed SGLang source: `/sgl-workspace/sglang`, commit
  `8eaffdf382e06b7dc50fc5c76cc5aad9c36bb0e4`.
- Delivery checkout: `/job/sglang`, PR base `main` at
  `0084030179bfba86bfeb6d43f7997d4076329d2c`.
- Checkout Python module: `/job/sglang/python/sglang/__init__.py`.
- Checkout FP8 kernel module:
  `/job/sglang/python/sglang/kernels/ops/quantization/fp8_kernel.py`.
- Triton module: `/opt/venv/lib/python3.10/site-packages/triton/__init__.py`.
- AITER native module observed during import:
  `/sgl-workspace/aiter/aiter/jit/module_aiter_core.so`.
- Job-private cache root: `/tmp/sglang-cache-j-fb721cb33726`.

## Installed-source first GPU baseline

The first GPU objective was completed before cloning or editing the delivery
checkout. The raw artifact is `/job/baseline-first.json`; it is evidence for the
installed source only and is not proof for later checkout changes.

Command:

```bash
/opt/venv/bin/python /job/baseline_first.py
```

The baseline used the installed `w8a8_block_fp8_matmul` Triton kernel with
`M=64`, `N=128`, `K=256`, block size `[128, 128]`, and BF16 output. The
independent reference dequantized both FP8 operands to FP32 and used
`torch.matmul`.

- First synchronized kernel call: `3.9228256652131677` seconds.
- First execution from process start: `8.440338999032974` seconds.
- Bounded timing method: three warmups, then ten synchronized iterations with
  `time.perf_counter`; mean `5.8696605265140533e-05` seconds per iteration.
- Numerical result: maximum absolute error `0.0`, mean absolute error `0.0`,
  relative mean error `0.0`.
- Input scale bytes were unchanged before and after the direct kernel call.
  No post-load transform was involved in this installed-source baseline.

## Persistent checkout case

Test:
`test/registered/amd/test_fp8_dense_weight_lifecycle_gfx942.py`.

Command:

```bash
export PYTHONPATH=/job/sglang/python
export TRITON_CACHE_DIR=/tmp/sglang-cache-j-fb721cb33726/triton
export TORCHINDUCTOR_CACHE_DIR=/tmp/sglang-cache-j-fb721cb33726/inductor
export HF_HOME=/tmp/sglang-cache-j-fb721cb33726/hf
export PIP_CACHE_DIR=/tmp/sglang-cache-j-fb721cb33726/pip
/opt/venv/bin/python -m pytest -q \
  test/registered/amd/test_fp8_dense_weight_lifecycle_gfx942.py -s
```

Result: `1 passed, 3 warnings in 16.30 seconds`. After a skip-path safety
tweak, the final affected-test run also passed: `1 passed, 3 warnings in 15.44
seconds`.

The deterministic case uses a `128x256` checkpoint-format FP8 weight and a
`64x256` BF16 input. It feeds checkpoint `weight` and `weight_scale_inv`
tensors through the real weight loader, runs
`Fp8LinearMethod.process_weights_after_loading`, pre-quantizes the input with
per-token-group FP8, and executes the dispatched dense linear kernel through
`ColumnParallelLinear`.

Observed transforms and invariants:

- Checkpoint weight dtype `torch.float8_e4m3fn` becomes runtime
  `torch.float8_e4m3fnuz` on `gfx942`.
- Checkpoint `weight_scale_inv` is multiplied by exactly `2.0`; this compensates
  the same-bit `e4m3fnuz` value being half of `e4m3fn`.
- Dynamic block-FP8 `input_scale` is `None`; no static activation scale is
  invented.
- The per-token-group activation scale is byte-identical before and after the
  real kernel call.
- The independent reference dequantizes the FP8 input and checkpoint weight to
  FP32 and uses `torch.matmul`.
- The unchanged numerical gate is `torch.testing.assert_close` with
  `rtol=5e-2` and `atol=1e-1`.

The first checkout attempt failed before GPU execution because the shared test
fixture initialized distributed state but did not publish the new
`ParallelContext` configuration. The concrete error was:

```text
ValueError: config namespace 'parallel' not published
```

The new test therefore uses `get_parallel().override(tp_size=1)` for its TP1
fixture. This is a test-fixture boundary fix, not a production dispatch change.

## Boundaries

- No full model weights or another framework stack were downloaded.
- No synthetic GPU burn, unbounded loop, sleep loop, or repeated work was used.
- No node-wide state was modified.
- The tested commit is preserved on branch
  `amdpilot/j-fb721cb33726`.
- Left undone: static per-tensor FP8 activation scales and MXFP8 dense linear
  were not claimed as supported by this focused block-FP8 lifecycle test.
