# Investigation: DSPARK accumulated additive penalties

Upstream issue: https://github.com/sgl-project/sglang/issues/33493

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1860

## Finding

The prepared base still read `SamplingBatchInfo.acc_linear_penalties` in the
DFlash verify adjustment helper and the DSPARK no-adjustment predicate. That
attribute does not exist; overlap-mode additive penalties are stored in
`acc_additive_penalties`. Consequently, the verify block omitted accumulated
additive penalties, including the negative-infinity stop-token mask used for
`min_new_tokens`, and DSPARK could select its no-adjustment fast path.

The failing-before regression recorded in `raw/failing_before.log` measured both
effects: four assertions failed because the adjustment was absent and the no-op
predicate returned true. The correction reads the existing field and broadcasts
it over each position in the verify block. Independent cases cover a one-token
boundary, a multi-token block, two batch rows, negative infinity, dtype
conversion, composition with logit bias, and the no-adjustment fast path.

## Related changes checked

- https://github.com/sgl-project/sglang/pull/33531 is closed and contained this
  diagnosis as part of a substantially broader patch.
- https://github.com/sgl-project/sglang/pull/33869 is open and proposes the same
  narrow field-name correction. It was not merged into base commit
  `358c163250ad3b1f62939b01ce1314a0a31a0365`.

## Limits

The reported DeepSeek-V4-Flash command requires model weights and TP4, while the
job provides one gfx950 GPU. No full-model or distributed serving reproduction
was claimed. The gfx950 check validates only the affected tensor adjustment on
the actual GPU. Separately, current DFlash decode preparation does not visibly
use the ordinary/EAGLE `cumulate_penalty_output_tokens` hook; whether that makes
the minimum-token mask lift later than intended needs a qualified model-level
investigation and is intentionally not folded into this stale-field fix.
