# Independent review of PR 536

Candidate: `ce616d9aa4709d956eda8a306e952d9ac70a72e5`

Base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

Recommendation: **accept**. The candidate fixes both independently reproduced violations, and no counterexample was found in the directly runnable contract.

## Evidence

On the prepared base, `DSparkVerifyPlanner.schedule_layout` returned through the cached compact verify-all path even when `forced_budget_frac=0.2`; `_schedule_verify_lens` was never called. The same base raised `RuntimeError: shape '[48, -1]' is invalid for input of size 640` for the issue's 48-request graph tier.

At the exact candidate commit, the two focused test modules passed 37 tests and 18 subtests. An independent forced-fraction sweep passed through `HostConfidenceBudgetPlanner.compute_budget` and the real GPU `ScheduleVerifyLensTopk` path. Fractions `0.2`, `0.4666666666666667`, `0.7333333333333333`, and `1.0` produced total verify widths `86`, `137`, `188`, and `240`, rather than a uniform full width.

The reported packed geometry (48 requests, 575 real tokens, 640 graph tokens) and 23 independently generated ragged/padded layouts were run on an AMD Instinct MI355X. Candidate output matched a separately constructed CPU row-by-row reference exactly, including per-request multimodal deltas and zeroed graph padding.

The candidate's no-force condition still takes the original cached verify-all branch. No native code changed, so rebuilding FlyDSL or another native library was not applicable. Imports resolved to the source checkout under `/job/repo/python/sglang`, not an installed sglang wheel.

## Limitations

The prepared environment is one AMD MI355X with ROCm 7.2 and PyTorch 2.11.0, not two NVIDIA H20 GPUs with CUDA 13.0. The reported model checkpoints were unavailable. Consequently, the full server/profiler command, TP=2 behavior, and NVIDIA CUDA Graph integration were not executed. These environment limits do not hide a failed substitute smoke: the production planner, budget computation, GPU top-k scheduler, and `ForwardBatch.compute_spec_mrope_positions` paths were invoked directly.

Raw command output was preserved outside the revision-switching checkout at `/job/review-evidence-j-0b86ff3d3e21`.
