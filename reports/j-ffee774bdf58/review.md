# Independent review of PR 1437

Upstream issue: https://github.com/sgl-project/sglang/issues/35201

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1474

Candidate: https://github.com/amdpilot-org/sglang/pull/1437 at `ace5d6123d773bd6fd760375352226467c82628c`

Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

## Recommendation

Request changes. This is a substantive partial fix, not test-only hardening, but it does not fully resolve the original contract.

## Evidence

The image-prepared checkout initially matched the recorded base exactly. On that base, the reported allocation arithmetic reproduced exactly:

`4096 * align256(92992) * 4 = 1,526,726,656 bytes = 1.421875 GiB`.

The candidate regression collected with an import error on the base because the budget helper did not exist. At the exact candidate, the focused suite passed: 21 tests and 25 subtests.

An independent candidate probe used the issue dimensions and the reported 1.17 GiB free-memory point. With the default free-memory fraction of 0.2, the budget was 251,255,586 bytes and the candidate selected 674 rows, producing a 251,224,064-byte maximum logits chunk instead of the 1.421875 GiB full rectangle. This is direct evidence that the normal budgeted path addresses the reported allocation mechanism.

On the assigned AMD Instinct MI355X (`gfx950`), an independent fp32 Top-K fixture produced identical selected page sets for one full transform and row chunks of 1, 7, 16, and 36. This validates that the Python row slicing preserves the reduction invariant. It does not execute the changed CUDA DeepGEMM path.

## Blocking counterexample

`PagedIndexerMetadata._mqa_logits_budget` returns `None` whenever `use_prefill_cuda_graph` is true, or while any of the checked graph-capture modes is active. For the issue-sized matrix this leaves `rows_per_chunk=None` unless an unrelated SM120 cap applies, so the candidate retains the original single full logits allocation in graph-backed prefill. The code comment confirms this is intentional rather than an incidental test gap.

The row planner also returns one row when the budget is smaller than one aligned row. For the issue width, one row is 372,736 bytes; budgets from 1 through 372,735 bytes are therefore exceeded. That boundary is less relevant to the reported 1M-token configuration, but it means the claimed budget is not a hard allocation bound.

## Environment and native-code scope

Python imports resolved from `/job/repo/python`. The interpreter was `/tmp/amdpilot-repo-j-ffee774bdf58/venv/bin/python`, using Torch 2.11.0+rocm7.2 and HIP 7.2.26015. The machine exposed one AMD Instinct MI355X (`gfx950`), not the reported 4xH100 TP4 CUDA setup. DeepSeek-V4-Flash-0731 weights were unavailable. No native source changed between the recorded base and candidate, so no native rebuild was applicable.

Raw JUnit evidence was preserved outside the checkout in `/job/review-evidence-j-ffee774bdf58/` while revisions were switched.
