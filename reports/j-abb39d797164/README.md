# gfx942 DeepSeek-V4 top-k execution-contract probe

This report covers the distinct dtype, graph-address, and reuse follow-up for
sgl-project/sglang issue 37892. It does not repeat the original dispatch bug
trigger from amdpilot-org/sglang issue 224 and does not duplicate the open
selector-overflow fix in sgl-project/sglang PR 37625.

## Scope and environment

- Tested mirror base: `0084030179bfba86bfeb6d43f7997d4076329d2c`.
- GPU: one AMD Instinct MI300X, gfx942, ROCm `7.2.26015-fc0010cf6a`.
- Interpreter: `/opt/venv/bin/python`, Python `3.10.12`.
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`.
- Operator-specified image ID: `sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1`.
- v1 native dispatch: `torch.ops.sgl_kernel.deepseek_v4_topk_transform_512` in `/opt/venv/lib/python3.10/site-packages/sgl_kernel/common_ops.cpython-310-x86_64-linux-gnu.so`.
- v2 native dispatch: checkout JIT `sgl_kernel_jit_dpsk_v4_topk_v2` loaded from the job-private cache under `/tmp/sglang-cache-j-abb39d797164`.

The installed-source baseline is in `installed-baseline.json`. It is evidence for
the preinstalled environment only, not for later checkout changes. Its FP32 v1
AOT control passed the independent CPU `torch.topk` comparison with zero
sentinels. The installed v2 JIT failed because its older source included
`cooperative_groups.h` unconditionally on ROCm. The persistent mirror main
already guards that include with `#ifndef USE_ROCM`, so checkout v2 compiled and
ran normally.

## Results

`gfx942-results.json` contains the complete raw evidence. The bounded timing
method is two warmups followed by five CUDA-event launches; the median is
reported.

| Path | Batch | Logical length | Top-k | Median |
| --- | ---: | ---: | ---: | ---: |
| v1 AOT | 4 | 4096 | 512 | 0.032635 ms |
| v2 JIT | 4 | 4096 | 512 | 0.026381 ms |
| v1 AOT | 8 | 16384 | 1024 | 0.043100 ms |
| v2 JIT | 8 | 16384 | 1024 | 0.029027 ms |
| v1 AOT | 2 | 65537 | 2048 | unsupported |
| v2 JIT | 2 | 65537 | 2048 | 0.059698 ms |

All supported FP32 cases matched an independently copied CPU `torch.topk`
reference exactly, with zero sentinel values remaining. v1 top-k 2048 was not
forced through its documented 1024 limit.

### Negative dtype evidence

FP16 and BF16 are not supported execution representations for this operation.
Both paths rejected them before launch and left the complete sentinel-filled
output untouched:

- v1 AOT FP16/BF16: `RuntimeError: scores must be float32 with shape [B, max_seq_len]`.
- v2 JIT FP16/BF16: `RuntimeError` from the TensorMatcher, reporting the supplied dtype and allowed dtype `[float32]`.

The probe created each dtype tensor, cloned it to CPU as an independent
representation, computed the CPU reference, and then attempted the native call.
No dtype was cast or reinterpreted to make an unsupported variant pass.

### Graph and aliasing contracts

- v1 and v2 CUDA/HIP graph capture and replay preserved the static score,
  output, and v2 plan addresses.
- Replayed outputs matched the independent CPU reference with zero sentinels.
- Ragged v2 changed only the documented masked head columns ahead of a
  non-trivial window and produced no stray writes.

## Reproduction

Run from the mirror checkout with the qualified interpreter and a job-private
JIT cache:

```bash
SGLANG_JIT_CACHE_DIR=/tmp/sglang-cache-j-abb39d797164 \
PYTHONPATH=/job/sglang/python \
/opt/venv/bin/python reports/j-abb39d797164/gfx942_probe.py

SGLANG_JIT_CACHE_DIR=/tmp/sglang-cache-j-abb39d797164 \
PYTHONPATH=/job/sglang/python \
/opt/venv/bin/python -m pytest \
  python/sglang/kernels/aot/tests/test_topk.py::test_deepseek_v4_topk_transform_rejects_non_fp32 \
  python/sglang/kernels/aot/tests/test_topk.py::test_deepseek_v4_topk_transform_cuda_graph_replay \
  test/registered/kernels/ops/attention/test_topk_v2.py::test_topk_v2_rejects_non_fp32_scores \
  test/registered/kernels/ops/attention/test_topk_v2.py::test_topk_v2_cuda_graph_replay_reuses_static_addresses \
  -vv
```

The focused run passed all eight selected tests: six new contract tests plus
one existing v1 and one existing v2 numerical control. No model weights,
toolchain replacement, unbounded loop, synthetic burn, or node-wide state change
were used.

## Limits

- The AOT binary was not rebuilt; v1 exercised the qualified installed native
  module while v2 exercised the checkout JIT source.
- PR 37625 was consulted read-only and remains open. Its selector-overflow/tie
  coverage was not duplicated or re-run here.
- This result does not claim to resolve the original GB300 illegal-address
  reproduction; it records the documented gfx942 dtype, graph-address, and
  in-place aliasing contracts for the relevant top-k paths.
