# gfx942 FP8 attention operator control

## Result

The installed AITER unified-attention FP8 path passed the existing numerical
test on one assigned MI300X. The persistent mirror checkout also passed a new
bounded synthetic context sweep at 256, 1024, 4096, 8192, and 16384 tokens.
Every BF16, independently dequantized-FP8, and FP8-versus-BF16 comparison had
zero elements outside the existing tolerance and cosine similarity above
0.999986. The longest-context FP8-versus-dequantized mean absolute error was
lower than the shortest-context value; there was no progressive error growth
in this operator-level control.

This is not a full-model NIAH reproduction and does not establish that the
upstream full-model issue is fixed.

## Environment

- GPU: one AMD Instinct MI300X, gfx942, unique ID
  `0x439e01ac3221d888`, serial `692440003981`, node ID 4.
- Required image: `amdpilotv2/open-job-mi300:jit-config-readable-260909-banff5`,
  local image ID
  `sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1`.
- Python: `/opt/venv/bin/python` (3.10.12).
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`, imported from
  `/opt/venv/lib/python3.10/site-packages/torch/__init__.py`.
- Installed-source SGLang baseline commit: `8eaffdf382e06b7dc50fc5c76cc5aad9c36bb0e4`,
  imported from `/sgl-workspace/sglang/python/sglang/__init__.py`.
- Mirror PR base: `0084030179bfba86bfeb6d43f7997d4076329d2c`.
- AITER Python path: `/sgl-workspace/aiter/aiter/__init__.py`.
- AITER native path: `/sgl-workspace/aiter/aiter/jit/module_aiter_core.so`.
- `sgl_kernel` Python path:
  `/opt/venv/lib/python3.10/site-packages/sgl_kernel/__init__.py`.
- `sgl_kernel` native path:
  `/opt/venv/lib/python3.10/site-packages/sgl_kernel/common_ops.cpython-310-x86_64-linux-gnu.so`.

The installed-source baseline is labeled separately and is not proof for later
checkout changes. Its first useful GPU execution and numerical result were
recorded 104 seconds after task start in `/job/baseline-first.json`.

## Commands

Installed-source baseline:

```bash
/opt/venv/bin/python -m pytest -q \
  /sgl-workspace/sglang/test/registered/attention/test_aiter_fp8_q_unified_attention.py
```

Result: 3 tests and 4 subtests passed in 37.54 seconds; wrapper wall time was
41.679248 seconds.

Persistent-checkout focused control:

```bash
PYTHONPATH=/job/sglang/python /opt/venv/bin/python -m pytest -q -s \
  test/registered/attention/test_aiter_fp8_q_unified_attention.py::TestAiterFP8QUnifiedAttention::test_fp8_context_growth_matches_bf16_and_dequantized_references
```

Result: 1 test and 5 subtests passed in 19.24 seconds; wrapper wall time was
23 seconds.

Full focused file:

```bash
PYTHONPATH=/job/sglang/python /opt/venv/bin/python -m pytest -q -s \
  test/registered/attention/test_aiter_fp8_q_unified_attention.py
```

Result: 4 tests and 9 subtests passed in 18.98 seconds; wrapper wall time was
23 seconds.

Syntax check:

```bash
/opt/venv/bin/python -m py_compile \
  test/registered/attention/test_aiter_fp8_q_unified_attention.py
```

`ruff` is not installed in the qualified environment, so no formatter or lint
command was run. `git diff --check` passed.

## Control design

- Decode query length: 1.
- Batch: 4.
- Query heads: 16; KV heads: 1 (GQA 16:1).
- Head dimension: 256.
- Page size: 16.
- Context lengths: 256, 1024, 4096, 8192, and 16384.
- Synthetic q/k/v are seeded random tensors. Prefixes are slices of one maximum
  length tensor, so content is controlled while length changes. The test also
  checks that the shortest and longest BF16 outputs differ, preventing a false
  length-invariant control.
- FP8 uses `float8_e4m3fn` q/k/v with q, k, and v descale tensors. BF16 uses
  the original q/k/v.

References are independent float32 einsum/softmax/einsum calculations:

- BF16 reference: original BF16 q/k/v cast to float32.
- Dequantized reference: the actual FP8 q/k/v tensors multiplied by their
  respective scales, then cast to float32.
- Direct comparison: FP8 kernel output versus BF16 kernel output.

The unchanged numerical gates are:

- All output values finite.
- Reference mean absolute value greater than 0.25.
- Mismatch fraction less than 0.005, where mismatch is
  `abs(actual - expected) > 0.15 + 0.15 * abs(expected)`.
- Cosine similarity greater than 0.99.

Timing uses CUDA events with exactly 3 warmups and 5 measurements per mode and
length, and reports the median. It is bounded and measures only the attention
call between events. The small medians are noisy and are not a scaling claim.

## Raw results

`results.json` contains the complete per-length metrics and timing samples from
the full focused run. All mismatch fractions were exactly zero.

| Length | BF16 max abs / cosine | FP8 vs dequantized max abs / mean abs / cosine | FP8 vs BF16 max abs / mean abs / cosine | BF16 / FP8 median ms |
| ---: | --- | --- | --- | ---: |
| 256 | 0.0020946 / 0.9999989 | 0.0129967 / 0.0055338 / 0.9999893 | 0.0156250 / 0.0049856 / 0.9999867 | 0.130622 / 0.087121 |
| 1024 | 0.0020134 / 0.9999989 | 0.0119162 / 0.0080251 / 0.9999981 | 0.0117188 / 0.0072703 / 0.9999968 | 0.155560 / 0.128577 |
| 4096 | 0.0019827 / 0.9999989 | 0.0119677 / 0.0080803 / 0.9999982 | 0.0117188 / 0.0072956 / 0.9999974 | 0.150748 / 0.135634 |
| 8192 | 0.0019707 / 0.9999988 | 0.0119532 / 0.0075747 / 0.9999981 | 0.0117188 / 0.0067947 / 0.9999969 | 0.151109 / 0.135312 |
| 16384 | 0.0019648 / 0.9999990 | 0.0114538 / 0.0067595 / 0.9999979 | 0.0117188 / 0.0060277 / 0.9999965 | 0.131784 / 0.133187 |

## Upstream context and limitations

Upstream issue 36390 remains open. Its comments report that a newer public
ROCm stack and dedicated DSV4 backend did not reproduce the full-model failure,
and a separate two-node run passed factual and needle probes through 64K
tokens. Related change PR 32516 enables an opt-in CK bpreshuffle block GEMM on
gfx942, and issue 28685 reports a separate gfx950 blockscale GEMM defect.

This control tests only the AITER unified attention operator on gfx942. It does
not test DeepSeek-V4-Flash, block-FP8 linear/GEMM paths, CK bpreshuffle, MoE,
CUDA graphs, TP/EP communication, multi-GPU behavior, or NIAH retrieval. It
also does not generalize the gfx950 GEMM finding to gfx942. No upstream issue,
PR, or comment was posted or modified.

No model weights were downloaded, no framework stack was replaced, and no
node-wide state was modified. The job-private cache directory was
`/tmp/sglang-cache-j-37ef47a2b994`.
