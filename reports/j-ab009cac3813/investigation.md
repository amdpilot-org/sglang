# Hidden-state prefill offset investigation

Upstream issue: https://github.com/sgl-project/sglang/issues/32089

Mirror issue: https://github.com/amdpilot-org/sglang/issues/3337

## Finding

The prepared base (`358c163250ad3b1f62939b01ce1314a0a31a0365`) already contains the
reported fix. Upstream commit `a0b7bcf59289a6cf916fa5ee44e3cfa865a25f3d`
(`[Feature] Support return_hidden_states="last" (#30177)`, 2026-08-02) changed
the same path in two required ways:

- `scheduler.py` snapshots `extend_input_len_per_req` when either logprobs or
  hidden states are requested, protecting output processing from overlap-time
  request mutation.
- `batch_result_processor.py` invokes `_append_prefill_hidden_states` for every
  row-contributing request. The helper advances by `extend_input_len` before it
  decides whether to store that request's rows.

No production correction was added because duplicating this existing fix would
not be justified. Two focused regression cases were added to retain the exact
5-row non-requesting + 3-row requesting example and an independent cached/
chunked plus discarded-output boundary.

## Evidence

- `raw/regression_before.log`: the pre-fix algorithm returns rows 0, 1, 2 and
  fails the expected rows 5, 6, 7 assertion (expected exit code 1).
- `raw/unit_after.log`: the complete focused processor test file passes,
  including the two new regressions.
- `raw/gfx950_hidden_state_offsets.log`: the production helper executes with
  ROCm tensors on the assigned AMD Instinct MI350X (`gfx950`) and returns the
  expected slices for both cases.

## Limitation

This is a deterministic processor-level reproduction and GPU numerical check,
not a full HTTP/model reproduction. No model weights were available or needed
to establish the offset accounting defect and current fix. Consequently this
does not claim semantic model accuracy, another model architecture, or any
distributed/multi-node behavior.
