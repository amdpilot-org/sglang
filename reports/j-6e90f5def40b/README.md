# Independent review of PR 3351

Upstream issue: https://github.com/sgl-project/sglang/issues/36889

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/3330

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/3355

Candidate: https://github.com/amdpilot-org/sglang/pull/3351 at
`09008c9230386b98e10e1a2ae0766cb9ce495d86`

Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

## Recommendation

**Accept as test-only hardening; it does not itself fully resolve the original
issue.**

The candidate adds no runtime or native-code change. Its focused regression
accurately covers the actual resolver's warning-level diagnostic and cap
arithmetic. The same test passes on the recorded base because the base already
contains the warning-level remedy requested by the upstream issue. Thus this
is useful regression coverage for an already-fixed source path, not a
failing-before/passing-after production fix.

The issue-shaped resolver case reproduced on the recorded base: eight
configured requests, ten Mamba slots, and a five-slot ratio resolve to two,
with an actionable `WARNING`. The reporter's cited revision
`aa8c950a3df62b6642c4ea60a93a5e3eb1a1450e` also contains that warning. On the
exact candidate, its three tests and nine adjacent Mamba tests pass.
Independent cases confirmed floor division at 39 slots, exact/over capacity,
DP request conversion, a tighter token-capacity limit, and automatic request
limit handling.

## Remaining counterexamples and limitations

- `/server_info` retains the configured top-level `max_running_requests`; the
  resolved value is nested per scheduler as
  `effective_max_running_requests_per_dp`. A consumer reading only the former
  can still infer the wrong executable concurrency.
- The candidate injects a ratio of five into the real resolver. It does not
  exercise GLM-5.3-Flash/DFlash configuration derivation, load the named target
  or drafter, or validate model semantics.
- The reported 75% throughput change was not reproduced. The environment has
  one AMD Instinct MI355X (`gfx950`) with ROCm 7.2, not two DGX Spark GB10
  (`sm_121`) nodes with CUDA 13. Required model weights were unavailable.
- No source/native files changed, so a native rebuild was not applicable. The
  prepared interpreter imported SGLang from this checkout and Torch from the
  prepared ROCm environment.

Raw commands and outputs are retained under `raw/`.
