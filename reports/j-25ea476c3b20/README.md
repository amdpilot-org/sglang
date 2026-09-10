# MI300X fused-MoE tuner validation report

## Scope

This investigation covers read-only upstream context for sgl-project/sglang
issue 13363 and a bounded run of the existing Triton fused-MoE ROCm tuner
candidate set on one assigned AMD Instinct MI300X (`gfx942`). The mirror base
commit is `0084030179bfba86bfeb6d43f7997d4076329d2c`.

No upstream issue, pull request, or comment was modified. Open upstream PRs
reviewed for overlap included 24373, 32152, 34698, 34838, 34839, and 34840.
None already added the numerical output gate implemented here.

## Environment

- Qualified image: `amdpilotv2/open-job-mi300:jit-config-readable-260909-banff5`
- Local image ID: `sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1`
- Interpreter: `/opt/venv/bin/python` (Python 3.10.12)
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`
- GPU: one MI300X, unique ID `0x419542839a8ee54b`, serial `692440003992`, node 5, `gfx942`
- Installed SGLang source: `/sgl-workspace/sglang/python/sglang`, commit `8eaffdf382e06b7dc50fc5c76cc5aad9c36bb0e4` (dirty environment context)
- Installed native module: `/opt/venv/lib/python3.10/site-packages/sgl_kernel/common_ops.cpython-310-x86_64-linux-gnu.so`
- Installed AITER module: `/opt/venv/lib/python3.10/site-packages/sgl_kernel/_rocm_aiter.cpython-310-x86_64-linux-gnu.so`

## Installed-source baseline

The first useful GPU control was the existing
`test_moe_align_block_size_compare_implementations[32-128-8-160-True]` case. It
compares the native `moe_align_block_size` operator with an independent Triton
implementation using exact integer checks.

```bash
PYTHONPATH=/sgl-workspace/sglang/python /opt/venv/bin/python -m pytest \
  '/sgl-workspace/sglang/python/sglang/kernels/aot/tests/test_moe_align.py::test_moe_align_block_size_compare_implementations[32-128-8-160-True]' \
  -q --tb=short -p no:cacheprovider
```

Result: 1 passed. The first successful GPU process took 7.477943358 seconds
wall time; pytest reported 4.48 seconds for the initial cold case. The metadata
is preserved in `/job/baseline-first.json`. This is only an installed-source
baseline and is not proof for later checkout changes.

The existing fused-MoE correctness test was also run from the mirror:

```bash
PYTHONPATH=/job/sglang/python /opt/venv/bin/python -m pytest \
  /job/sglang/test/registered/moe/test_fused_moe.py::TestFusedMOE::test_various_configurations \
  -q --tb=short -p no:cacheprovider
```

Result: 1 passed, 384 subtests passed, 73.94 seconds reported by pytest
(78.446731357 seconds process wall time).

## Candidate measurements

The tiny candidate set came from the existing
`get_rocm_configs_compute_bound()` search space. All three candidates used
`num_stages=2`, `GROUP_SIZE_M=1`, `num_warps=4`, and `waves_per_eu=0`:

| BLOCK_M | BLOCK_N | BLOCK_K | Correct | Mean time |
|---:|---:|---:|---|---:|
| 32 | 16 | 32 | yes | 16.057450 µs |
| 64 | 32 | 64 | yes | 16.884959 µs |
| 128 | 64 | 128 | yes | 19.599210 µs |

Shape: 8 tokens, 8 experts, hidden size 64, shard intermediate size 128, top-k
2, bfloat16. Correctness was checked against an independent native PyTorch
expert-loop with SiLU gate/up and top-k weighted sum before timing. Timing used
a CUDA graph containing 10 invocations, 5 warmup replays, 10 timed replays,
CUDA events, and an L2 cache flush before each replay.

The modified `benchmark_config` integration measured 15.932360 µs for the
winning candidate with the same validation and timing method. No generated
tuning JSON was committed or installed into the repository's shared config
directory; all generated files remained under the job-private
`/tmp/sglang-cache-j-25ea476c3b20` cache.

## Validation behavior

The new focused GPU test runs a valid candidate through the independent
reference and then mocks an invalid fused-MoE output. The valid candidate passes
and the invalid output raises `AssertionError`, so it cannot be timed or become
a tuning winner. Result: 1 passed, 14.50 seconds reported by pytest.

Default tuning now validates each candidate before timing. Validation failures
are skipped like compile/resource failures. If no candidate is valid,
`best_config` remains unset and the existing assertion prevents a winner.

## Limitations

- `ray` is not installed in the qualified image, so the full Ray-based CLI could
  not run. A minimal no-op import stub was used only to exercise the real
  `benchmark_config` function directly on the assigned GPU.
- `flashinfer` is not installed, and the current FlashInfer autotune path is
  CUDA-gated; it is not a viable MI300X candidate in this stack.
- Numerical validation is implemented for the unquantized standard SiLU fused
  MoE path. Quantized paths and the DeepSeek V4 swiglu-limit path require the
  explicit `--skip-output-validation` opt-out, which documents that invalid
  candidates may then win.
- Validation uses `0.01`-scaled copies to keep bfloat16 reference comparisons
  bounded; the original unit-variance tensors remain unchanged for timing.
