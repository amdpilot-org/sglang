# GLM-5.3 qkvbfg fusion investigation

Upstream issue: https://github.com/sgl-project/sglang/issues/38253

Mirror issue: https://github.com/amdpilot-org/sglang/issues/764

Base commit: `358c163250ad3b1f62939b01ce1314a0a31a0365`

## Finding

The bug is present at the recorded base. `Glm5NextLinearAttention` used
`quant_config is None` to decide whether to construct the qkvbfg fusion. That
tests whether any part of the model is quantized, rather than whether the six
source projections are quantized. Consequently an FP8 checkpoint whose
attention projections are excluded from quantization still selected the
unfused path.

The closed, unmerged upstream PR https://github.com/sgl-project/sglang/pull/37744
described the same cause and supplied checkpoint observations from the official
GLM-5.3-Flash weights. Its change was not present in this checkout.

## Correction

The fusion decision now resolves the six real projection prefixes through the
active quantization configuration. Fusion is allowed only if every prefix maps
to `None` or `UnquantizedLinearMethod`, and the existing attention-TP condition
still holds. The synthetic fused projection is constructed without the global
quantization configuration because its synthetic prefix is not checkpoint
metadata and the gate has already established that all source weights are
unquantized.

## Evidence

- `raw/reproduce-before.txt`: the actual base gate evaluates false even though
  all six prefixes resolve to `UnquantizedLinearMethod` (expected assertion
  failure, exit 1).
- `raw/failing-before.txt`: the regression added before implementation also
  failed collection because the issue-specific decision function did not exist
  yet (pytest exit 2).
- `raw/focused-tests-after.txt`: the issue configuration and two independent
  boundaries pass (partially quantized attention remains unfused; attention TP
  mismatch remains unfused), along with the existing skip-prefix tests (7
  passed).
- `raw/gfx950-fused-linear-numerical.txt`: on the assigned AMD Instinct MI350X
  (`gfx950`), the actual BF16 `MergedColumnParallelRepeatedLinear` and
  `ColumnParallelBatchedLinear` outputs exactly matched independent PyTorch
  `linear`/`bmm` references (both maximum absolute errors were 0.0).
- `raw/pre-commit.txt`: repository pre-commit checks passed for all changed
  Python files.

## Limitations

The official GLM-5.3-Flash weights were not available and the assignment had a
single GPU, so the original full-model TP8 serving, semantic accuracy, and
performance claims were not reproduced. The GPU check validates only the fused
linear primitives used by the corrected path. No native C++ source changed and
no native rebuild was required.
