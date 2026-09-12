# Correction generation 1: graph-aware default admission

Reviewed candidate: https://github.com/amdpilot-org/sglang/pull/1913 at
`a3ff51e96cf96008a8382e5e18ec0af420519a17`.

Independent review: https://github.com/amdpilot-org/sglang/pull/1983.

The candidate's one-time startup warning is a valid diagnostic improvement, but
it leaves the reported defaults unchanged: the issue fixture still resolves
3,145,118 KV tokens and context length 32,768 to 4,096 running requests while
decode graph coverage ends at 32. It therefore does not prevent eager fallback.

This correction reconciles the defaults by capping only the automatically
derived per-worker request limit at the enabled decode graph maximum. An
explicit `--max-running-requests` remains authoritative and retains the
candidate warning when it exceeds graph coverage. Disabled graphs and graph
configurations without a maximum retain the capacity-derived behavior.

## Evidence

- Exact candidate test: 3 passed.
- Exact candidate test against recorded base: 3 failed with `AttributeError`,
  reproducing the independent review's test-quality counterexample.
- New behavioral regression against recorded base: fails with the meaningful
  assertion `4096 != 32`.
- New regression and adjacent pool-configurator suite after correction: 47
  passed and 7 subtests passed.
- `git diff --check`: passed.

Commands and concise raw outputs are retained under `raw/`.

## Limitations

The prepared host exposes AMD gfx950, not the reported NVIDIA L40, and the
Qwen2.5-0.5B weights and ShareGPT workload were not available. No claim is made
that the reported 4x latency trajectory was reproduced here. The source-level
policy defect and its numeric counterexample are deterministic; full L40/Qwen
performance validation remains outstanding. No native code changed.
