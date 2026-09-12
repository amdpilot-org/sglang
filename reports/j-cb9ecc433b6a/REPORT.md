# Independent review of amdpilot-org/sglang PR 832

Reviewed exact candidate commit `0622ecefe18513f109ee381dd2974c9d0d6d50ae`
against recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`.

Recommendation: **request changes**.

The candidate's touching-segment coalescing fixes the original issue's direct
failure. On the recorded base, actual `free_kv_row_segments` cleanup of
`[0, 5)` followed by `[5, 7)` reaches `_page_disjoint` and raises
`AssertionError: segment at 5 shares a page with the one ending at 5`. At the
exact candidate, the same probe frees the two unique physical pages once.

The correction is not fully idempotent, however. Its completed-free fingerprint
contains `kv_indices._version`. Request rows are views of the single shared
`ReqToTokenPool.req_to_token` tensor, so a write to unrelated request row B
increments the version observed by fresh slices of row A. A subsequent identical
retry of A no longer matches its request-local completed record and frees A's two
pages again. The independent GPU probe observed free-page deltas `(2, 2)` rather
than `(2, 0)` after A, an unrelated row-B write, then retry A.

Candidate regression coverage passed (`55 passed, 41 subtests passed`) and the
original A/B/A counterexample without an intervening table mutation passed with
deltas `(2, 1, 0)`. Independent cases for three touching segments plus a gap,
request-local separation, and failed-call non-memoization also passed on the
assigned GPU.

No C++ or other native source changed. Imports resolved to the candidate checkout
under `/job/repo/python`; no native rebuild was applicable. The prepared machine
has one AMD Instinct GPU with ROCm 7.2 and PyTorch 2.11.0, not eight NVIDIA B300
GPUs (SM103), and it has no Kimi-K3 or DSpark weights. Therefore the full
multi-rank production serving replay is unverified; the evidence covers the real
Python allocator cleanup path on GPU.

Raw logs and standalone probes are retained outside the checkout at
`/job/review-evidence-j-cb9ecc433b6a/`.

