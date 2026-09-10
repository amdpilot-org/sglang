# DSA top-k v2 output reuse investigation

## Result

On one MI300X (`gfx942`), fresh output storage and safe reuse of one
sentinel-protected output buffer both matched an independent `torch.topk`
selected-index reference across two distinct input batches. Every output slot
was overwritten: no `-12345` sentinel survived, valid counts and `-1` padding
matched each row length, and the reused output address remained stable.

The production plan buffer also kept its address after an in-place refresh.
On ROCm the plan kernel is intentionally compiled out, so this run verifies
address stability but not refreshed plan contents; the registered test checks
content refresh only on non-HIP runtimes.

Before this change, an output view exactly aliasing the page table passed the
shape and dtype checks and dispatched. The host wrappers now reject output
aliasing with `topk output must not alias an input`. Unsupported `int64`
outputs and `float16` scores already failed clearly through `TensorMatcher`
and remain unsupported.

## Reproduction

Use the qualified interpreter and a job-private JIT cache:

```bash
export PYTHONPATH=/job/sglang/python
export SGLANG_JIT_CACHE_DIR=/tmp/sglang-cache-j-fa24c8498610
/opt/venv/bin/python -m pytest \
  test/registered/kernels/ops/attention/test_topk_v2.py \
  -q -k output_storage_reuse
```

The unchanged numerical gate was:

```bash
/opt/venv/bin/python -m pytest \
  test/registered/kernels/ops/attention/test_topk_v2.py -q
```

It passed all 279 cases. The timing matrix used three warmups and ten measured
calls for each `{fresh, reused} x {batch 1, 2, 4}` cell, including allocation
or sentinel refill, kernel execution, and synchronization. Raw values are in
`results.json`.

## Evidence boundaries

The preinstalled source at `/sgl-workspace/sglang` could not build the v2 JIT
module because `cooperative_groups.h` was missing under its ROCm 7.2 include
configuration. That first execution failed after 25.327 seconds. A bounded
`torch.topk` control ran on the same input sizes; its raw timings are also in
`results.json`. The installed-source result is environment context only.

The mirror checkout at base `0084030179bfba86bfeb6d43f7997d4076329d2c`
contains the ROCm include guard and built successfully. Upstream PR 37941
was consulted as the existing threshold-bin exactness candidate but was not
applied or duplicated; this work covers the distinct output-reuse and ABI
case.
