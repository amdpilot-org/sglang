# Independent review of PR 1917

Reviewed https://github.com/amdpilot-org/sglang/pull/1917 at exact commit
`3a1c517def5a96209f259fabe8df8cb4e8cbc85c` against:

- Upstream issue: https://github.com/sgl-project/sglang/issues/34112
- Mirror issue: https://github.com/amdpilot-org/sglang/issues/1953

## Recommendation

Request changes. The candidate is a verified partial source fix, not a verified
full resolution of the original issue.

The candidate's regressions were copied onto recorded base
`358c163250ad3b1f62939b01ce1314a0a31a0365` before its source changes were
applied. Three issue-specific checks failed: the cancelled final-prefill request
committed token/metadata, the beam variant did not cleanly consume the pending
abort, and the request entered optimistic decode. On the exact candidate, its
focused scheduler/result-processing suite passed with 23 tests and 2 subtests.

The source change gives a pending final-prefill abort precedence over sampled
token, metadata, grammar, and beam-prefill commits, and excludes such a request
from optimistic decode while a delayed result is queued. This directly repairs
the visible-token path in the deterministic fixtures.

The original report has a second observable contract: later scheduler probe
records must stop containing negative `output_ids`. Neither candidate test file
asserts scheduler output-ID values, and the candidate's own report says the
negative-ID symptom was not reproduced or ruled out. Therefore the PR's
`outcome: fixed` claim is broader than its evidence. A regression using the
original probe, or an equivalent deterministic test of the recorded scheduler
field, is still required before calling the original issue fully resolved.

## Environment and paths

Python imports on the exact candidate resolved to the checkout's `sglang`,
`scheduler.py`, and `batch_result_processor.py`. The prepared AITer module
loaded from the private runtime cache.

The prepared interpreter used Torch `2.11.0+rocm7.2`. One assigned AMD
Instinct MI350X reported `gfx950:sramecc+:xnack-` and completed a tensor sum of
`5.0`. This is architecture/environment evidence only; it is not a serving
reproduction. The issue's NVIDIA RTX 4090/CUDA environment and
`meta-llama/Llama-3.2-1B-Instruct` weights were unavailable. The qualified tiny
Llama fixture cannot establish model-specific semantics or substitute for the
missing original workload.

No native source changed. `repository-environment.json` records `native: null`,
so no native rebuild was applicable.

Raw outputs are summarized in `evidence/raw-output.txt`. The downloaded issue
reproducer, source issue snapshot, candidate metadata/diff, and complete test
logs were preserved outside the checkout under
`/job/review-evidence-j-13cccda2c293/` while revisions were switched.
