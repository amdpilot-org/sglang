# Independent review of PR 3063

Reviewed candidate: https://github.com/amdpilot-org/sglang/pull/3063 at exact commit `512cdcfcf8b16bb55f328a84068c6b82f77a6c80`.

Upstream issue: https://github.com/sgl-project/sglang/issues/34899

Mirror issue: https://github.com/amdpilot-org/sglang/issues/3091

## Verdict

Request changes. The candidate is useful test-only hardening and fixes the
specific averaged-threshold oracle defect, but it does not fully qualify the
original unified-memory feature request.

On the recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`, unequal
logprobs `[0.0]` and `[-1e-5]` produce nonzero KL
`5.0000069648240756e-11` and pass the `1e-9` averaged threshold. At the exact
candidate, the new `require_exact=True` path rejects that case and also rejects
a `1e-12` numerical difference. Its three focused oracle unit tests pass.

The candidate adds six collected unified-memory cases: baseline,
prefill-cache-hit, and decode-cache-hit for both Python and Rust tree backends.
They are not independently qualified end to end here. The assigned GPU is an
AMD Instinct MI355X (`gfx950`) under ROCm 7.2. Inkling imports its NVIDIA CUTE
attention path and fails because `cutlass` is unavailable. An actual launch of
the Python unified-memory baseline consequently failed during model
registration before an inference request or KL measurement. The Rust backend
shares the same unsupported model prerequisite and was not redundantly
launched.

No runtime or native source changed, so no native rebuild was applicable. The
active imports on both revisions resolved to the checkout:
`/job/repo/python/sglang/__init__.py` and
`/job/repo/python/sglang/test/kl_test_utils.py`.

The exact comparator is value-exact rather than byte-exact: NumPy accepts
positive versus negative zero and equal values with different dtypes. Those
cases have zero KL and therefore do not contradict the issue's explicit “any
nonzero KL” oracle, but the implementation should not be described as literal
byte-for-byte equality without qualification.

## Evidence summary

- Base reported counterexample: exit 0; nonzero KL accepted.
- Base collection: exit 0; 18 tests, no unified-memory class.
- Candidate oracle unit suite: exit 0; 3 passed.
- Candidate collection: exit 0; 24 tests, including six unified-memory cases.
- Candidate adversarial oracle: reported and `1e-12` differences rejected;
  signed-zero and dtype-only equal values accepted.
- Candidate Inkling direct import: exit 1, `ModuleNotFoundError: cutlass`.
- Candidate Python unified-memory baseline launch: exit 1 before inference;
  model module import was ignored after the missing-CUTLASS error, then model
  registration failed.

Complete raw command output is preserved outside the revision-switched checkout
under `/tmp/amdpilot-repo-j-c249d356ff54/review-evidence/`.

## Remaining qualification work

1. Run baseline, prefill-cache-hit, and decode-cache-hit with the shrunken
   Inkling checkpoint on supported hardware for both Python and Rust trees and
   record exact-zero logprob/KL results.
2. Demonstrate revert-then-red or targeted fault injection for unified-memory
   FULL, SWA, and MAMBA restoration boundaries.
3. Keep the candidate's exact-oracle correction; collection and unit-level
   comparator tests alone are not proof that the restoration tests detect the
   defects claimed by the original issue.
