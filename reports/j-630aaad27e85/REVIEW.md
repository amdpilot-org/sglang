# Independent review of candidate `8f262970bde402bb73e2fb05ff9f6b7a93c3c74b`

Upstream issue: https://github.com/sgl-project/sglang/issues/21443

Mirror issue: https://github.com/amdpilot-org/sglang/issues/3221

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/3007

Candidate PR: https://github.com/amdpilot-org/sglang/pull/3034

Parent candidate PR: https://github.com/amdpilot-org/sglang/pull/2901

Parent independent review: https://github.com/amdpilot-org/sglang/pull/2972

## Verdict

Recommendation: **accept**. The exact candidate fully resolves the original
source-level memory-reporting contract and the concrete counterexamples from
the parent review.

On recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`, an independent
probe supplied 64 GiB host total and 60 GiB host available. All three reported
MPS paths returned those unsafe host values: SRT available 60 GiB, device-stub
total 64 GiB, multimodal total 64 GiB, and multimodal available 60 GiB.

At exact candidate `8f262970bde402bb73e2fb05ff9f6b7a93c3c74b`, the candidate's
23 tests passed. An independent adversarial probe verified all three production
integrations, Metal-limited and host-pressure-limited cases, exact-limit and
over-limit allocations, cache-empty behavior, and fail-closed behavior for
missing, raising, non-positive, non-numeric, and infinite API results. The
registered-test taxonomy check and the complete changed-file pre-commit run
also passed.

The implementation does not set an MLX wired-memory limit. That is not a
remaining counterexample for these PyTorch/MPS paths: the issue's explicit
expected behavior is capped memory reporting, while the wired-limit call is
part of the cited MLX reference implementation. The candidate consistently
caps the three named reporting paths using PyTorch's Metal-backed
`recommended_max_memory()` and current driver allocations.

## Environment and architecture limitations

The prepared host is Linux/x86_64 with Torch 2.11.0+rocm7.2. Torch reports MPS
unavailable. Consequently, real `MTLDevice.recommendedMaxWorkingSetSize`
values, Apple UMA pressure/paging behavior, machine stability, and an actual
MPS model/KV-cache workload were not exercised. AMD GPU execution would not
validate Apple Metal accounting, so no GPU test was claimed. The change is
Python-only; no native source changed and no native rebuild was applicable.

The source/import probe resolved `sglang` to
`/job/repo/python/sglang/__init__.py` and Torch to the prepared environment at
`/opt/venv/lib/python3.12/site-packages/torch/__init__.py`. Raw revision-specific
evidence was preserved outside the checkout in
`/job/review-evidence-j-630aaad27e85/` before switching back to the review
branch.
