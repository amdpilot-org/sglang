# Investigation report

Upstream issue: https://github.com/sgl-project/sglang/issues/34974

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1460

Base commit: `358c163250ad3b1f62939b01ce1314a0a31a0365`

The prepared checkout still contained the defect. `DSparkV4Stage._run_ffn` invoked
the draft MoE without a target-layer recorder context. When EPLB recording or
graph-capture hooks were active, the shared target recorder forwarded
`layer_idx=None` to `_SelectExpertsSinglePassGatherer`; indexing its 2-D count
tensor with `None` produced a 2-D destination while the expert IDs were 1-D.
The preserved pre-fix run reaches the reported `scatter_add_` dimension error
through the checked-out DSpark method and real recorder implementation.

The fix disables the target recorder only around the draft MoE call. DSpark
stages own separate NextN weights and zero-based stage IDs, so attributing their
traffic to target-layer rows would corrupt the statistics used by EPLB. The
existing nested, exception-safe `disable_this_region()` context provides the
narrow ownership boundary.

Upstream PR https://github.com/sgl-project/sglang/pull/37947 independently
contains this same candidate but was still open when investigated. Its issue
comments include reporter validation on the original 8xH20 production shape.
This checkout did not already contain that solution.

## Evidence

- `raw/regression-before-exact.txt`: failing-before reproduction, including the
  exact reported exception in recording-active and capture-active modes.
- `raw/regression-after.txt`: passing-after regression and independent
  exception/nested-context boundary cases.
- `raw/neighbor-regression.txt`: neighboring DSpark projection tests.
- `raw/gfx950-capture-after.txt`: actual HIP graph capture/replay on the assigned
  AMD Instinct MI350X (gfx950), with exact output and target-count checks.
- `gfx950_capture_check.py`: reproducible GPU seam check.

No DeepSeek-V4-Flash weights were available and only one gfx950 GPU was
assigned. Therefore this work does not claim a full-model, H20, TP=8/EP=8,
semantic-accuracy, acceptance-rate, multi-node, or live-migration validation.
No native component changed or required rebuilding.
