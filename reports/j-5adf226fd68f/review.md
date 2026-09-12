# Independent review of PR 1748

Candidate: https://github.com/amdpilot-org/sglang/pull/1748 at
`6bd589c04122574293c44fa322dd765f43133bfe`

Upstream issue: https://github.com/sgl-project/sglang/issues/35201

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1785

Parent candidate: https://github.com/amdpilot-org/sglang/pull/1608 at
`c0d54de7cd2aafe30f38d9b98c4c8e9c034ec70c`

Parent independent review: https://github.com/amdpilot-org/sglang/pull/1687

## Recommendation

Accept. The candidate fixes the concrete remaining graph-budget counterexample
without changing native code. At the issue shape, the recorded base dispatches
the complete logits rectangle in one call. The rectangle is
`4096 * align(92992, 256) * 4 = 1,526,726,656` bytes (1.421875 GiB), larger
than the issue-reported 1.17 GiB free memory. Candidate commit `6bd589c` caps
the graph-safe budget at 512 MiB and plans 1,440 rows per chunk, about 500 MiB.
The candidate also preserves eager live-free-memory tightening, chunks the
non-paged CUDA DeepGEMM path, builds matching schedule and Top-K metadata for
each paged chunk, and explicitly errors when even one aligned row cannot fit.

The original issue's unbounded context-proportional allocation contract is
therefore resolved in the reviewed source: the per-call logits allocation is
bounded rather than scaling without limit with query rows. No independent
source counterexample remained after boundary testing.

## Evidence

- Prepared base: `358c163250ad3b1f62939b01ce1314a0a31a0365` (the prepared checkout
  matched the recorded base).
- Exact candidate: `6bd589c04122574293c44fa322dd765f43133bfe`.
- Source imports resolved to `/job/repo/python/sglang/...`; Torch resolved to
  the prepared ROCm environment.
- Candidate issue regression: `graph_budget=536870912`,
  `issue_allocation=1526726656`, `rows_per_chunk=1440`.
- Focused DSV4 tests: 21 passed, 29 subtests passed.
- Shared DSA budget tests: 3 passed.
- Independent adversarial cases covered graph/eager budgets, 100 MiB live-free
  memory, `mem_fraction_static` values of 0.5, 0.9, 1.0, and unset, the
  below-one-row diagnostic, exact/full chunk boundaries, and ragged GPU Top-K.
- On the assigned AMD Instinct MI350X, chunk sizes 1, 3, 7, 16, 36, 37, and
  100 selected exactly the same pages as an unchunked 37x4096 fp32 transform.
- Python compileall and `git diff --check` passed.

Raw evidence is retained outside the checkout at
`/job/review-evidence-j-5adf226fd68f/` so revision switching did not overwrite
it.

## Limitations

The environment has one AMD Instinct MI350X (`gfx950`) with ROCm 7.2, not four
H100 GPUs with CUDA 12.9. DeepSeek-V4-Flash-0731 weights were unavailable.
Consequently the exact 800K-token TP4 serving workload and CUDA DeepGEMM kernel
execution were not run. The GPU check establishes row-wise Top-K equivalence,
not model semantics, distributed behavior, or CUDA/H100 execution. No native
source changed in the candidate, so no native rebuild was applicable.
