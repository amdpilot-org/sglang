# Independent review of amdpilot-org/sglang PR 2938

Reviewed exact candidate commit `7da983b52f8e95df6c41bebe4a685775e3e65f9f`
against the recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`.

Upstream issue: https://github.com/sgl-project/sglang/issues/33035

Candidate PR: https://github.com/amdpilot-org/sglang/pull/2938

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2979

Recommendation: **request changes**. The candidate fixes the previously reported
budget-one/32-body counterexample by rejecting 31 requests before ASGI `receive`,
and its focused tests pass. It does not fully implement weighted admission during
the body stage: every unparsed request reserves one item. With an item budget of
4, four concurrent bodies whose eventual cost is four items each were all read
and retained before their actual weights could be known; weighted admission would
permit only one. The original Kimi-K2.6 workload remains unverified, and the
option explicitly rejects the Rust frontend and EPD language-only mode.

No native source changed. Python imports were confirmed to resolve from this
checkout. GPU execution and a native rebuild were not applicable to this CPU
HTTP admission review.

Raw command output was preserved outside revision switching at
`/job/review-evidence-j-369ce08d0c96/` in the review environment.
