# gfx942 block-FP8 GEMM execution-representation control

## Result

The installed supported gfx942 path was exercised first. On this checkout,
`aiter_w8a8_block_fp8_linear` routes the non-bpreshuffle gfx942 case to AITER's
Triton block-FP8 GEMM, not CK. The new focused test validates that uncovered
execution representation for BF16 and FP16 outputs against an independent
float32 dequantized reference, uses a NaN-sentinel output buffer, checks output
aliasing, captures and replays one CUDA graph, and records a bounded timing
matrix.

All four focused GPU tests and six subtests passed. BF16 and FP16 outputs were
finite, every NaN sentinel was overwritten, the returned tensor was the supplied
`y` storage, the output did not alias either input, and one graph replay produced
bit-identical output at the same address. Unsupported input representations now
fail with clear Python exceptions before native dispatch.

This is an operator-level control. It does not reproduce or claim to fix the
full-model DeepSeek-V4-Flash behavior in upstream issue 36390.

## Environment

- GPU: one AMD Instinct MI300X, gfx942, unique ID
  `0x439e01ac3221d888`, serial `692440003981`, node ID 4.
- Required image: `amdpilotv2/open-job-mi300:jit-config-readable-260909-banff5`,
  local image ID
  `sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1`.
- Python: `/opt/venv/bin/python` (3.10.12).
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`, imported from
  `/opt/venv/lib/python3.10/site-packages/torch/__init__.py`.
- Installed-source SGLang baseline commit:
  `8eaffdf382e06b7dc50fc5c76cc5aad9c36bb0e4`, imported from
  `/sgl-workspace/sglang/python/sglang/__init__.py`.
- Mirror PR base: `0084030179bfba86bfeb6d43f7997d4076329d2c`.
- AITER Python path: `/sgl-workspace/aiter/aiter/__init__.py`, commit
  `c16d44b93a528b2a4bfd6d8d3409116d465872a9`.
- AITER Triton wrapper:
  `/sgl-workspace/aiter/aiter/ops/triton/gemm/basic/gemm_a8w8_blockscale.py`.
- Profiled native kernel:
  `_gemm_a8w8_blockscale_kernel_GROUP_K_128_GROUP_N_128_BLOCK_SIZE_M_128_BLOCK_SIZE_N_128_BLOCK_SIZE_K_128_GROUP_SIZE_M_1_NUM_KSPLIT_1_SPLITK_BLOCK_SIZE_7168_EVEN_K_1_GRID_MN_4_cache_modifier_CG`.

The installed-source baseline is recorded separately in
`/job/baseline-first.json`. It is not proof for later checkout changes. Its
first useful GPU execution completed 11.228 seconds after that process started.
An initial run without `SGLANG_USE_AITER=1` failed with
`NameError: name 'triton_gemm_a8w8_blockscale' is not defined`; the supported
neighboring control with the environment variable set dispatched to the Triton
path and passed the independent-reference check.

## Commands

Installed-source baseline:

```bash
SGLANG_USE_AITER=1 /opt/venv/bin/python /job/baseline_first.py
```

Persistent-checkout focused GPU control:

```bash
SGLANG_USE_AITER=1 \
TRITON_CACHE_DIR=/tmp/sglang-cache-j-c15fafde1fc6/triton \
PYTHONPATH=/job/sglang/python \
/opt/venv/bin/python -m pytest -q -s \
  test/registered/unit/layers/test_fp8_blockscale_gemm_mi300x.py
```

Result: 4 tests and 6 subtests passed in 15.60 seconds.

Existing gfx95 coverage collection:

```bash
SGLANG_USE_AITER=1 PYTHONPATH=/job/sglang/python \
/opt/venv/bin/python -m pytest -q \
  test/registered/unit/layers/test_fp8_bpreshuffle_dense_linear_mi35x.py
```

Result: 2 tests skipped on gfx942, as expected.

Syntax and whitespace checks:

```bash
/opt/venv/bin/python -m py_compile \
  python/sglang/srt/layers/quantization/fp8_utils.py \
  test/registered/unit/layers/test_fp8_blockscale_gemm_mi300x.py
git diff --check
```

Both passed. `ruff` is not installed in the qualified environment, so no
formatter or lint command was run.

## Control Design

- Weight shape: `N=512`, `K=7168`, block size `[128, 128]`.
- Token matrix: `M=1`, `128`, and `1024`.
- Output dtypes: BF16 and FP16.
- Input and weight values are independently represented as native FP8 tensors
  with explicit float32 scales. The reference dequantizes both operands to
  float32, applies the scales, and performs a float32 matmul before casting to
  the requested output dtype.
- The output buffer is filled with NaN before each direct GEMM call. The test
  requires the returned object and address to be the supplied `y`, all sentinels
  to be overwritten, and the output not to alias either GEMM input.
- The CUDA-graph test warms up once, captures one call, replaces the static
  input in place, replays once, and requires the same output address and
  bit-identical output to a fresh direct call.
- Timing uses exactly 3 warmups and 5 CUDA-event measurements per dtype and M;
  it is evidence only and is not a pass/fail gate or scaling claim.

Numerical gates are unchanged in intent: all outputs finite, no NaN sentinels
remaining, and `torch.testing.assert_close` with `rtol=2e-2` and
`atol=2e-5` against the independent reference.

## Raw Results

`results.json` contains all six numerical records and timing samples. Median
GEMM times ranged from 0.194 ms (`M=1`, BF16) to 0.392 ms (`M=1024`, BF16).
The largest absolute reference error was `4.76837158203125e-07`; mean absolute
errors were at or below `2.1763372354266508e-11`.

The direct AITER Triton helper accepted an exploratory FP32 output despite its
docstring limiting `dtype` to BF16 or FP16. That behavior is not used or forced
through the SGLang wrapper. The wrapper now rejects unsupported float32 input
before native quantization, and rejects a non-native-FP8 input when a prequantized
input scale is supplied.

## Upstream Context And Limitations

Upstream issue 36390 remains open. Its comments report that a newer public ROCm
stack and dedicated DSV4 backend did not reproduce the full-model failure, and a
separate two-node run passed factual and needle probes through 64K tokens.
Related PR 36396 added MI300X DeepSeek-V4-Flash FP8 accuracy coverage.

This control covers only one MI300X, one block-FP8 GEMM shape family, direct
BF16/FP16 output representations, and the SGLang wrapper's prequantized BF16
path. It does not test DeepSeek-V4-Flash, CK bpreshuffle, MoE, TP/EP, multi-GPU
communication, full-model NIAH retrieval, or the gfx950 blockscale defect from
issue 28685.

No model weights were downloaded, no framework stack was replaced, and no
node-wide state was modified. Generated Triton artifacts were kept in
`/tmp/sglang-cache-j-c15fafde1fc6/triton`. No upstream issue, PR, or comment was
posted or modified.
