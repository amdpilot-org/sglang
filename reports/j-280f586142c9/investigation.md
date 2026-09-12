# Independent review of PR 1333

Reviewed https://github.com/amdpilot-org/sglang/pull/1333 at exact commit
`f487356650d0657f192b8544b04da645f810ed6b` against:

- Upstream issue: https://github.com/sgl-project/sglang/issues/37712
- Mirror issue: https://github.com/amdpilot-org/sglang/issues/1365

## Finding

Recommendation: **request changes**. The candidate is a useful partial fix, but
it does not fully resolve the original availability contract.

The recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365` computes one
`n_real * total_k_rows` fp32 logits tensor. On the assigned GPU, the candidate's
regression suite fails 8 of 9 tests on that base. In particular, the independent
grouped fixture observes a 240-byte allocation against a 24-byte budget.

The exact candidate groups K rows by request and chunks Q rows, eliminating the
cross-request amplification that explains the reported 73.65 GiB allocation.
It also rejects the prior review's atomic-row counterexample before calling
DeepGEMM: a 10-column fp32 row requires 40 bytes and cannot be allocated under a
4-byte budget. The candidate's full focused suite passes (12 tests), and an
independent exact/adjacent-boundary suite passes (4 tests) on real gfx950 GPU
tensors.

However, the over-budget atomic-row case now raises `RuntimeError`. It avoids an
allocation beyond the chosen budget, but it does not compute a result or preserve
service availability. Under the same condition on all TP ranks, the DSA path can
still abort on every rank. A complete fix needs a K-splittable logits/top-k path
with globally correct merging, or explicit request-scoped error handling proven
not to terminate the workers. The candidate correctly acknowledges that missing
capability, so this is a partial source fix rather than a full original-issue fix.

## Environment and source verification

`sglang` and `dsa_indexer_kpool` imported from the checked-out source under
`/job/repo/python`, not from an installed copy. Testing used one AMD Instinct
MI355X (`gfx950`) with PyTorch 2.11.0+rocm7.2 and HIP 7.2.26015. The reported
four-B300 CUDA TP=4/EP=4 deployment, GLM-5.3-Flash/RadixArk weights,
hierarchical cache, DFLASH traffic, and NVIDIA DeepGEMM kernels were unavailable.
DeepGEMM and fused top-k were mocked by the focused tests; therefore the original
serving workload and end-to-end numerical semantics remain unverified.

No C++/HIP/CUDA/native source changed between the recorded base and candidate.
Consequently no native rebuild was applicable. The loaded AIter extension was
the prepared environment artifact; it was not used as proof of the candidate.

## Evidence

- `base-regression.log`: candidate regression tests run against recorded base;
  8 failed, 1 passed, demonstrating the pre-fix allocation behavior.
- `candidate-environment.log`: exact source import paths and gfx950 environment.
- `candidate-full-regression.log`: exact candidate; 12 passed.
- `independent-budget.log`: independent 40-byte/4-byte and exact/adjacent budget
  cases; 4 passed.
