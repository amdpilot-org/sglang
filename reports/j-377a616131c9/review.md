# Independent review of amdpilot-org/sglang PR 824

Upstream issue: https://github.com/sgl-project/sglang/issues/38253

Mirror issue: https://github.com/amdpilot-org/sglang/issues/862

Candidate: https://github.com/amdpilot-org/sglang/pull/824 at
`fce408041e990861151b72d113404cbe194b3333`

Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

## Recommendation

Accept. The candidate fully resolves the original reported gate defect for the
documented GLM-5.3-Flash FP8 configuration: when the model has a global FP8
configuration but the linear-attention subtree is excluded, all six source
projections resolve as unquantized and qkvbfg fusion is enabled. It preserves
the attention-TP layout guard and rejects fusion if any source projection is
quantized.

This is a source fix with regression hardening, not merely a test change. No
native source changed, so no native rebuild was applicable.

## Evidence

On the prepared base, `reproduce_before.py` resolved all six source prefixes to
`UnquantizedLinearMethod` but the actual base predicate
`quant_config is None and 8 == 8` was false. The expected assertion failed with
exit code 1; see `raw/base-reproduction.txt`.

After temporarily checking out the exact candidate, imports resolved to the
checkout under `/job/repo/python/sglang`, as recorded in
`raw/candidate-import-paths.txt`. The candidate's focused regression plus the
existing quantization-prefix suite passed 7 tests.

Independent boundary checks established:

- the official-style parent exclusion enables fusion;
- checkpoint leaf exclusions (`q_proj`, `k_proj`, `v_proj`, and the five other
  projection leaves) enable fusion;
- a missing projection, an unrelated exclusion, and attention-TP mismatch all
  keep fusion disabled;
- a model without a quantization configuration retains the prior fused path.

On the assigned AMD Instinct MI350X (`gfx950`), the actual
`MergedColumnParallelRepeatedLinear` and `ColumnParallelBatchedLinear` classes
ran in BF16. Independent PyTorch `linear` and `bmm` references matched exactly.
A second check loaded all eight source matrices through the real weight-loader
methods before comparing the two-stage fused computation; every maximum
absolute error was 0.0. See `raw/gpu-fused-primitives.txt` and
`raw/gpu-loader-and-math.txt`.

## Counterexample outside the original configuration

An ignore list written with the internal fused module name `qkv_proj` plus the
other five internal module prefixes does not enable fusion, because existing
`is_layer_skipped` logic expands `qkv_proj` to the checkpoint shard names and
requires `q_proj`, `k_proj`, and `v_proj`. This is a conservative false negative,
not unsafe fusion, and it does not match the official parent-prefix shape or
checkpoint leaf naming described by the original issue. The candidate's own
partial-projection test does not expose this distinction. Details are in
`raw/prefix-resolution-detail.txt` and `raw/adversarial-prefix-cases.txt`.

## Limitations

The official GLM-5.3-Flash weights were unavailable. Only one gfx950 GPU was
assigned. Therefore this review does not claim full-model serving, TP8 or
multi-node reproduction, semantic accuracy, or a performance result. The GPU
evidence qualifies the changed fused primitives and their loader behavior, not
the complete model architecture. No C++/CUDA/HIP/FlyDSL file changed, and the
prepared environment declared no separate native artifact, so
`native_rebuilt` is false because rebuilding was not applicable.
