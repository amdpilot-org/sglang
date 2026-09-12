# Independent review of PR 1750

Candidate: https://github.com/amdpilot-org/sglang/pull/1750 at
`d0fbb177b80d57956c3fd543e5f70a306b69f4ad`

Upstream issue: https://github.com/sgl-project/sglang/issues/34943

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1788

Recorded and prepared base: `358c163250ad3b1f62939b01ce1314a0a31a0365`
(the prepared checkout did not differ from the recorded base).

## Verdict

Recommendation: **accept**. The candidate addresses both coupled parts of the
original contract: speculative verify/draft batches are classified as decode,
and torch trace compression/export is moved off the scheduler thread. It also
removes the background collective-ordering hazard found in the prior review and
makes profile-v2 export asynchronous. This is a source fix with regression
coverage, not test-only hardening.

## Failing before

The candidate's regression was saved outside the checkout and run against the
prepared base. `TARGET_VERIFY` mapped to `prefill`; the legacy predicate
misclassified `TARGET_VERIFY` and rejected `DRAFT_EXTEND_V2`. The default and
profile-v2 nonblocking tests each failed after blocking for five seconds inside
`export_chrome_trace`, demonstrating that the caller could not continue.

## Passing candidate

At the exact candidate commit:

* `test_profiler_manager_stage_stop.py`: 5 passed (including two parameterized
  speculative-mode subtests).
* Existing profile merger/API tests: 13 passed.
* The retained staggered two-rank Gloo case completed with exit 0; both ranks'
  normal `all_reduce` returned `3.0` while exports were staggered. No export
  thread issued a distributed collective.
* Independent real-GPU checks exercised both the legacy manager and profile-v2
  on the assigned AMD Instinct MI350X/gfx950. Each stop returned while its
  export thread was still alive, the thread subsequently terminated, and a
  nonempty gzip trace was produced (41,810 and 41,668 bytes respectively).
  The 256x256 GPU matmul maximum absolute errors versus CPU were
  `2.86102294921875e-05` and `2.6702880859375e-05`.

Imports resolved to `/job/repo/python/sglang/...`; Torch resolved to the
prepared ROCm environment (`2.11.0+rocm7.2`). No C++/HIP/native source changed,
so a native rebuild was not applicable.

Raw logs and the review fixtures were preserved outside the checkout at
`/job/review-evidence-j-7f2c1b4f8edd/` while revisions were switched.

## Limitations

Only one gfx950 GPU was available. B200 x8, GLM-5.2 weights, TP8/MTP serving,
the reported 242 MB trace, the exact 25-second pause, and end-to-end TTFT were
not reproducible in this environment. The two-rank Gloo case validates CPU
process-group ordering rather than multi-rank GPU profiling. These limitations
prevent reproducing the production scale and duration, but the issue's two
causal contracts were directly reproduced on the base and independently tested
on the candidate. No remaining source-level counterexample was found within the
approved scope.
