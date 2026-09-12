# Independent review of PR 1634

Upstream issue: https://github.com/sgl-project/sglang/issues/34384

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1668

Candidate: https://github.com/amdpilot-org/sglang/pull/1634 at
`0a0b787c4dbc878065c48926d18c5b43eafcf19a`.

## Recommendation

Request changes. The candidate is test-only hardening, not a production fix.
The prepared base already contains a plausible production solution: replay pads
the live ragged layout to the captured request-slot count, caps rows to the
configured verify width, stages `verify_lens` and `qo_indptr` into stable
capture buffers, and DSV4 rebuilds replay metadata from the padded view.

The candidate's two new tests validate only
`RaggedVerifyLayout.padded_to_bucket`. They do not call graph admission,
`DecodeCudaGraphRunner._stage_ragged_verify_layout`, replay metadata refresh,
or the DSV4 backend. Consequently, they would still pass if the integration
behavior that addresses the original illegal access were removed. An
integration-level regression should assert that the 32-by-6 live layout is
staged into the captured 192-slot buffers and consumed by replay metadata.

## Evidence

- On the recorded base, the pre-existing ragged-layout suite passed 10 tests
  and scheduler boundary tests passed 2 tests. This means the candidate's tests
  are not failing-before evidence for a newly fixed defect.
- At the exact candidate, the expanded suite passed 12 tests and the scheduler
  boundary tests passed 2 tests.
- An independent gfx950 device test invoked the real runner staging method.
  The reported 32 requests by 6 tokens produced 192 stable capture slots with
  `[6] * 32 + [0] * 160`; a mixed-width 33-request layout, a 160-real-token
  layout rounded to the 192-token tier, and the native 192-request boundary
  also passed. Shrinking 192 live rows to 191 slots was rejected.
- Imports resolved to `/job/repo/python/sglang`, using the prepared interpreter
  and Torch 2.11.0+rocm7.2. No native source changed, so no native rebuild was
  applicable.

## Limitations

The original failure was not reproducible end to end here. The assigned host
has one AMD Instinct MI355X (`gfx950`) with ROCm 7.2, not four NVIDIA H20 GPUs,
and the DeepSeek-V4-Flash DSpark checkpoint was unavailable. The tests qualify
layout construction and staging on one GPU, not CUDA/Hopper graph replay,
DSV4 model semantics, TP4, or the reported full serving workload. Therefore
the claim that the original issue is fully resolved remains unverified even
though no source-level geometry counterexample was found on the available
hardware.
