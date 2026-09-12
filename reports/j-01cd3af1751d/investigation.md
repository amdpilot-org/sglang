# Independent review of PR 1922

- Upstream issue: https://github.com/sgl-project/sglang/issues/34676
- Mirror issue: https://github.com/amdpilot-org/sglang/issues/1961
- Candidate: https://github.com/amdpilot-org/sglang/pull/1922
- Exact candidate commit: `a91b8669fd1c60e0fa98ee81dabb8e78a93c85ca`
- Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

## Recommendation

Accept. The candidate fully resolves the original issue's scheduler-availability
contract in the exercised implementation: a late KV-cache allocation miss is
typed, newly acquired request-pool state is rolled back, scheduler ownership is
restored, and an idle scheduler can retry instead of being permanently gated by
`batch_is_full`. Existing active chunk ownership and nonempty decode-batch
backpressure are retained.

The recorded base reproduced the original control-flow failure: an injected
late allocation `RuntimeError` escaped `_get_new_batch_prefill_raw`. At the exact
candidate commit, the focused regression suite passed (39 tests and 12
subtests). Independent cases also showed that a transient miss succeeds on the
next pass without duplicate queue ownership, and that an unrelated
`RuntimeError` is not swallowed.

## Evidence

- `raw/base_repro.txt`: base exception escapes from `prepare_for_extend`.
- `raw/candidate_focused.txt`: candidate regression and PrefillAdder suite.
- `raw/adversarial.txt`: transient retry and narrow exception-boundary checks.
- `raw/import_paths.txt`: source checkout imports and runtime/device identity.

## Limitations

The original eight-NVIDIA-B300 TP8/DCP4 Kimi-K3 hybrid-Mamba workload and model
weights were unavailable. The prepared host exposes one AMD Instinct MI350X
(gfx950) through PyTorch 2.11.0+rocm7.2. The deterministic tests validate the
real Python scheduler/allocation paths, but not CUDA/B300 behavior, distributed
execution, model semantics, or sustained production traffic. No native source
changed, so no native rebuild was applicable. GPU availability was inventoried,
but the issue-specific tests did not execute GPU kernels.
