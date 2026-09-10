# gfx942 structured-input INT8 block-GEMM study

## Scope

This study follows mirror issue 407 and read-only upstream context from sgl-project/sglang issue 15194. It is distinct from mirror PR 475, which compared random-input AITER and Triton INT8 backends. This run keeps one supported installed kernel and varies complete activation blocks across structured distributions.

The tested kernel is `sglang.kernels.ops.quantization.int8_kernel.w8a8_block_int8_matmul`, backed by the Triton JIT kernel `_w8a8_block_int8_matmul`. No kernel or dispatch code is changed.

## Environment

- GPU: one AMD Instinct MI300X, `gfx942:sramecc+:xnack-`, 304 compute units
- Image: `amdpilotv2/open-job-mi300:jit-config-readable-260909-banff5`, local image ID `sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1`
- Python: `/opt/venv/bin/python`
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`
- Source path: `/job/sglang/python/sglang/kernels/ops/quantization/int8_kernel.py`
- Native path: `/tmp/sglang-cache-j-fc5321c2589b/triton/PQADCGMG5WN24VC3RXIFCC2UM2NDFRNCYXI6XRVJIR34OUOG4OAQ/_w8a8_block_int8_matmul.hsaco`
- Source commit: `0084030179bfba86bfeb6d43f7997d4076329d2c`

## Installed-source baseline

Before cloning or editing, the installed source at `/sgl-workspace/sglang` ran the same Triton INT8 block GEMM for one `M=1`, `N=1024`, `K=1024` mixed-magnitude case. The independent float32 dequantized reference matched exactly. The bounded timing method used three warmups and ten CUDA-event samples, with a mean of `0.05072090029716492 ms`. First GPU execution, including Triton compilation, took `1.5283266715705395` seconds.

The complete baseline record is outside the repository at `/job/baseline-first.json`. It is installed-source evidence only and is not proof for the later checkout.

## Workload matrix

All cases use `N=1024`, `K=1024`, `[128, 128]` blocks, packed INT8 operands, float32 scales, and bfloat16 output. The synthetic weight is 1,048,576 bytes. Peak allocated memory during the checkout run was 152,176,128 bytes.

| Case | M | Input distribution |
|---:|---:|---|
| 1 | 1 | 90% zeros and 10% tiny ±1 values with `1e-6` activation scales |
| 2 | 1 | Mixed INT8 magnitudes with per-block scales from `1e-6` to `1e-2` |
| 3 | 1 | Alternating ±127 cancellation rows |
| 4 | 4096 | 90% zeros and 10% tiny ±1 values with `1e-6` activation scales |
| 5 | 4096 | Mixed INT8 magnitudes with per-block scales from `1e-6` to `1e-2` |
| 6 | 4096 | 25% alternating ±127 cancellation rows and 75% skewed sparse rows |

The weight has a structured all-ones first `N` block. Alternating activation rows therefore cancel exactly in that block while other blocks exercise nonzero mixed contributions.

## Numerical contract

Every complete output tensor is compared with an independent float32 dequantized matmul and then cast to bfloat16. The unchanged gate from the existing INT8 block test is:

```text
mean(abs(actual - reference)) / mean(abs(reference)) < 0.02
```

All outputs must be finite. All six cases pass. The largest mean-relative error is `4.0136956158676185e-7`; the largest absolute error is `0.03125` in the large-`M` mixed-magnitude case.

## Timing

Timing uses three warmups and 20 CUDA-event samples per case. It is bounded evidence on shared hardware, not a pass/fail performance gate.

| M | Distribution | Median ms | CV |
|---:|---|---:|---:|
| 1 | zero/tiny | 0.058556 | 0.098954 |
| 1 | mixed magnitude | 0.057052 | 0.050376 |
| 1 | cancellation/skewed | 0.057513 | 0.091893 |
| 4096 | zero/tiny | 0.073509 | 0.041929 |
| 4096 | mixed magnitude | 0.074231 | 0.046774 |
| 4096 | cancellation/skewed | 0.073390 | 0.049793 |

The complete measurement run took 4.281800684984773 seconds.

## Reproduce

```bash
cd /job/sglang
export PYTHONPATH=$PWD/python
export TRITON_CACHE_DIR=/tmp/sglang-cache-j-fc5321c2589b/triton
/opt/venv/bin/python -m pytest -q -s \
  test/registered/unit/layers/quantization/test_int8_block_gemm_structured_inputs.py
/opt/venv/bin/python reports/j-fc5321c2589b/run_structured_int8_study.py \
  reports/j-fc5321c2589b/results.json
```

The qualified test run passed one test and six subtests in 34.31 seconds. The measurement run completed in 4.28 seconds.

## Limits

- This is one MI300X and one `N=K=1024` shape family.
- It tests the SGLang Triton INT8 path only; it does not compare AITER or other backends.
- Timing variation on shared hardware is reported with each median and coefficient of variation.
- No model weights, MoE path, multi-GPU path, or production-model claim is covered.
- No upstream issue, PR, or comment was posted or modified.
