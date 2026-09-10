# MI300X compiled MoE combine reuse investigation

This is a bounded follow-up to amdpilot-org/sglang issue 231 and read-only context
from sgl-project/sglang issue 13363. It does not repeat issue 231's tuner-winner
validation. The uncovered case is the small-token MoE combine's
`torch.compile` execution representation and its reuse across a finite shape
sequence.

## Result

The installed-source control was the unquantized BF16 Triton fused-MoE path on
one AMD Instinct MI300X (`gfx942`). It used an independent fp32 expert-loop
reference and three CUDA-event timed calls for each of token counts 1, 3, 7,
and 16. All input and weight sentinels remained unchanged and every output was
finite. Raw metadata and measurements are in `/job/baseline-first.json`; that
installed-source result is not evidence for later checkout changes.

The mirror experiment exercised `moe_sum_reduce_torch_compile` with BF16, top-k
2, hidden size 64, and token counts 1, 2, 4, 8, 16, and 32. It used an
independent fp32 `sum(dim=1) * scale` reference, a NaN-filled output sentinel
before the cold call and every warm call, and the same output allocation for
cold and warm calls. All numerical checks passed with zero absolute error, the
input remained unchanged, every sentinel was overwritten, and the output data
pointer and storage pointer stayed stable.

The first shape paid 932.587 ms for compilation. The second shape paid
98.224 ms while Dynamo generalized the shape; the remaining four cold calls
were already warm and took 0.124-0.150 ms. Warm medians were 0.078-0.117 ms.
The profiler recorded the actual runtime dispatch as
`triton_poi_fused_copy__mul_sum_0` followed by `hipModuleLaunchKernel`; it did
not dispatch through the eager `aten::sum`/`aten::mul_` path. Raw timings,
addresses, checks, dispatch events, and errors are in `results.json`.

## Contract fix

The unguarded helper allowed `torch.sum(..., out=...)` to resize a mismatched
output and accepted a mismatched output dtype. Both behaviors can invalidate a
caller's static graph-address assumption. The helper now validates rank, shape,
matching float32/float16/bfloat16 dtype, device, contiguity, and non-overlap
before entering the compiled implementation. Mismatched dtype, mismatched
shape, and overlapping input/output storage therefore fail with `ValueError`
rather than being forced through.

## Reproduction

```bash
PYTHONPATH=/job/sglang/python /opt/venv/bin/python -m pytest \
  /job/sglang/test/registered/unit/layers/moe/test_moe_sum_reduce_compile_reuse.py \
  -q --tb=short -p no:cacheprovider

PYTHONPATH=/job/sglang/python /opt/venv/bin/python \
  /job/sglang/reports/j-8389ea6de1bb/run_reuse_experiment.py \
  --output /tmp/moe-combine-reuse-results.json
```

The experiment performs 6 cold calls and 18 warm calls, plus one profiled call.
It performs no cache reset between shapes, uses no model weights, and does not
run an unbounded loop.

## Environment

- Image: `amdpilotv2/open-job-mi300:jit-config-readable-260909-banff5`
- Local image ID: `sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1`
- GPU: one AMD Instinct MI300X, `gfx942`, UUID `31633363-3932-6564-6537-623264636230`
- Interpreter: `/opt/venv/bin/python`
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`
- Triton: `3.7.0`
- Mirror base commit: `0084030179bfba86bfeb6d43f7997d4076329d2c`
- Tested code commit: `b7d72ea09072d4e82742c0416eedfa11f8fadcf8`

## Limitations

- Timing is a small single-GPU matrix, not a shared performance database.
- The experiment covers BF16, top-k 2, hidden size 64, and token counts through
  32, matching the helper's documented small-token use.
- The installed `sgl_kernel` Python package exposes `moe_sum_reduce`, but its
  native op is not registered in this image, so the experiment records the
  compiled path's actual dispatch rather than comparing to that unavailable
  native entry point.
- No upstream issue, pull request, or comment was posted or modified.
