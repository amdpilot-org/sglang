# Independent review of PR 1280

Reviewed `https://github.com/amdpilot-org/sglang/pull/1280` at exact commit
`75d9c94e2f76938cdffe2777a52d28389b847bdf` against:

- Upstream issue: https://github.com/sgl-project/sglang/issues/36886
- Mirror issue: https://github.com/amdpilot-org/sglang/issues/1314

## Recommendation

Accept the candidate. It corrects the concrete regression left by the prior
candidate: the no-rope DCP writer now applies `reserved_skip_index` before
owner selection/localization, matching the rope path and preserving the public
default reserved slot. It also retains the earlier candidate's index-K virtual
capacity and capacity-assert fixes.

This review does **not** mark the whole original serving issue fully verified.
The prepared machine has one AMD MI350X (`gfx950`) under ROCm, while the report
requires GLM-5.3-Flash weights and an 8xH100 DCP deployment to exercise the
watermark, sparse decode/verify LSE merge, sparse extend gather, and semantic
quality workload end to end.

## Independent evidence

The recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365` was checked
out first. An independent GPU fixture used DCP world size 2, both simulated
ranks, rope and no-rope inputs, a NaN source at virtual loc 0, owned and
non-owned virtual locations, and both the public default skip and explicit
`reserved_skip_index=-1`.

- Base, default skip: exit 1. The no-rope path overwrote physical slot 0 with
  NaNs instead of preserving the reserved row.
- Base, skip disabled: exit 1. The no-rope path failed owner/local-index
  semantics.
- Candidate, all eight combinations: exit 0. Default slot preservation,
  `loc // 2` localization, non-owner suppression, rope parity, and explicit
  loc-0 writes with skipping disabled all passed.
- Candidate regression selection: 6 passed.
- Complete MLA buffer test module: 11 passed, 58 CUDA-TMA-only tests skipped.
- DSA virtual-capacity unit tests: 3 passed.

Raw command output and the independent fixture are retained outside the
checkout at `/job/review-evidence/j-9f6f8ce6f884/` so revision switching could
not alter the evidence.

## Source and native-path checks

Both revisions imported
`/job/repo/python/sglang/kernels/ops/kvcache/mla_buffer.py`. The candidate only
changes Python/Triton kernel code plus Python configuration/backend code and
tests; it changes no C++, HIP, CUDA, or FlyDSL native source, so a separate
native rebuild was not applicable. Triton 3.7.0 compiled and executed the
changed no-rope kernel on the assigned MI350X through torch 2.11.0+rocm7.2.

The candidate contains trailing whitespace in several committed historical
test-output logs, so `git diff --check` is not clean. This is report-artifact
hygiene rather than a functional defect in the candidate source.

## Limitations and residual risk

No actual remaining counterexample was found in the corrected scatter
contract. The following original-issue scenarios remain unverified here:

- 8-rank H100 DCP execution and collective behavior.
- GLM-5.3-Flash full-model semantic output before and after crossing the
  cumulative-allocation watermark.
- End-to-end sparse decode/target-verify page-table localization and LSE merge.
- End-to-end sparse-extend RAGGED transform and sharded-prefix all-gather.
- CUDA TMA-only execution paths, which are unsupported on the ROCm host.
