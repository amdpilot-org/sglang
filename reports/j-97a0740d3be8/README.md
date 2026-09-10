# gfx942 FP8 attention page and block partition control

## Result

No partition-dependent output mismatch was demonstrated on one assigned MI300X
(`gfx942`). For identical finite adversarial FP8 inputs, page sizes 1, 2, 4, 8,
16, 32, and 64 produced bit-identical outputs at the tested decode shape. A
deterministically permuted block table that preserved logical page order also
produced bit-identical output relative to the identity block table.

Every partition case remained finite and matched an independently derived
float32 reference with mismatch fraction 0, cosine similarity 1.0, maximum
absolute error 0.13336181640625, and mean absolute error 0.03301245719194412.
The unchanged numerical gates were:

- all output elements finite;
- mismatch fraction strictly less than 0.005, using
  `abs(actual - expected) > 0.15 + 0.15 * abs(expected)`;
- cosine similarity strictly greater than 0.99.

Because no mismatch was demonstrated, no production code was changed.

## Scope

This control is separate from the already-open context-length control in
`amdpilot-org/sglang` pull request 323. That PR varies sequence length at a
fixed page size. This investigation varies page size and block-table placement
while holding the logical FP8 inputs fixed.

Read-only upstream context was taken from `sgl-project/sglang` issue 36390.
The issue remains open. Its linked change,
`e9bf815e53af7eabf321fc044ea492e3a0b21809`, only switched an AMD CI job to the
cookbook’s verified MI300X recipe; it did not fix or test page/block partition
stability.

## Environment

- GPU: one AMD Instinct MI300X, `gfx942:sramecc+:xnack-`, UUID
  `34333965-3031-6163-3332-323164383838`.
- Required image: `amdpilotv2/open-job-mi300:jit-config-readable-260909-banff5`,
  local image ID
  `sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1`.
- Python: `/opt/venv/bin/python` (3.10.12).
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`, imported from
  `/opt/venv/lib/python3.10/site-packages/torch/__init__.py`.
- Installed-source SGLang baseline commit:
  `8eaffdf382e06b7dc50fc5c76cc5aad9c36bb0e4`, imported from
  `/sgl-workspace/sglang/python/sglang/__init__.py`.
- Persistent mirror PR base: `0084030179bfba86bfeb6d43f7997d4076329d2c`.
- AITER Python path: `/sgl-workspace/aiter/aiter/__init__.py`.
- AITER native path: `/sgl-workspace/aiter/aiter/jit/module_aiter_core.so`.
- `sgl_kernel` Python path:
  `/opt/venv/lib/python3.10/site-packages/sgl_kernel/__init__.py`.

The installed-source baseline is labeled separately in
`/job/baseline-first.json` and is not proof for later checkout changes.

## Commands

Installed-source baseline:

```bash
timeout 180s /opt/venv/bin/python -m pytest -q \
  /sgl-workspace/sglang/test/registered/attention/test_aiter_fp8_q_unified_attention.py
```

Result: 3 tests and 4 subtests passed in 40.43 seconds; wrapper wall time was
44.375 seconds.

Partition probe:

```bash
PYTHONPATH=/job/sglang/python:/tmp \
  timeout 180s /opt/venv/bin/python /tmp/partition_probe.py
```

The temporary probe used one finite adversarial matrix with values drawn from
`{0, ±112, ±224, ±448}`, batch 4, 16 query heads, 1 KV head, head dimension
256, and sequence length 4096. It quantized q/k/v to `float8_e4m3fn`, reshaped
the same logical tensors for each page size, and compared every output with an
independently derived float32 einsum/softmax/einsum reference.

Block-table permutation control:

```bash
PYTHONPATH=/job/sglang/python:/tmp \
  timeout 120s /opt/venv/bin/python - <<'PY'
# Deterministically permute physical pages while preserving logical order,
# then compare identity and permuted block-table outputs directly.
PY
```

The direct identity-versus-permuted comparison had maximum absolute error 0,
mean absolute error 0, mismatch fraction 0, and cosine similarity 1.0.

## Raw Results

Single-shot elapsed times include Triton JIT/autotune and are not benchmark
comparisons.

| Page size | Elapsed ms | vs reference max abs | vs reference mean abs | Mismatch fraction | Cosine |
|---:|---:|---:|---:|---:|---:|
| 1 | 906.116 | 0.133362 | 0.033012 | 0 | 1 |
| 2 | 2.585 | 0.133362 | 0.033012 | 0 | 1 |
| 4 | 1384.190 | 0.133362 | 0.033012 | 0 | 1 |
| 8 | 448.956 | 0.133362 | 0.033012 | 0 | 1 |
| 16 | 2.810 | 0.133362 | 0.033012 | 0 | 1 |
| 32 | 442.912 | 0.133362 | 0.033012 | 0 | 1 |
| 64 | 963.385 | 0.133362 | 0.033012 | 0 | 1 |

Every page-size output was also bit-identical to the page-size 16 baseline.

## Boundaries

- No unsupported page size was encountered for 1–64 at this shape.
- Page size 0 is outside the operation contract and was not tested.
- The probe covers one decode shape and one finite adversarial input matrix; it
  does not establish invariance for every attention shape or full-model workload.
- No full model weights were downloaded, no environment was replaced, and no
  artificial GPU burn or unbounded loop was used.
