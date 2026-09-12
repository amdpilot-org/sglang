# Investigation notes

Upstream issue: https://github.com/sgl-project/sglang/issues/38167

Mirror issue: https://github.com/amdpilot-org/sglang/issues/794

At base `358c163250ad3b1f62939b01ce1314a0a31a0365`, the diffusion scheduler's
execution exception handler converted every exception to an `OutputBatch` and
continued its event loop. Thus the implementation had no guard against the
reported follow-up request after CUDA reported either an unavailable device or
an allocator invariant failure.

The correction recognizes only those two reported signatures. It sends the
triggering request an explicit restart-required error and sets `_running` false,
which lets the normal reply and cleanup path complete without accepting another
request. It does not classify ordinary OOM as device poisoning and does not
claim that the first error was necessarily caused by memory exhaustion.

Raw issue snapshots and test output are retained in `raw/`. The original CUDA
12 GB MiniMax-H3 workload was unavailable; the assigned ROCm GPU was used only
for the independently referenced numerical device-execution check.
