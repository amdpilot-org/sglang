# Independent review of PR 2496

Reviewed candidate: https://github.com/amdpilot-org/sglang/pull/2496 at
`7497fc4126053b57a6bfd978656697c0ddd858ab`

Upstream issue: https://github.com/sgl-project/sglang/issues/28887

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/2468

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2530

## Verdict

Recommendation: **accept**.

The candidate is valid test-only hardening, not the production fix. The recorded
base `358c163250ad3b1f62939b01ce1314a0a31a0365` already uses
`getattr(route, "path", request.url.path)`, so the original `_IncludedRouter`
`AttributeError` is already prevented before the candidate's changes. The
candidate adds a focused regression that uses FastAPI's real path-less wrapper,
plus direct-route and unmatched-route boundaries.

The regression was independently shown to fail when the helper was temporarily
restored to the vulnerable `return route.path, True` expression. It failed at
the helper with exactly `AttributeError: '_IncludedRouter' object has no
attribute 'path'`. After restoring the exact candidate, all three focused tests
passed and a real request through SGLang's Prometheus middleware returned HTTP
200.

Additional nested-router, integer-converter, query-string, method-mismatch, and
unmatched-converter cases passed. No native source changed, so no native rebuild
was applicable. Imports resolved to the checkout's
`/job/repo/python/sglang/srt/utils/common.py`; FastAPI 0.141.1 and Starlette 1.6.0
resolved from the prepared interpreter's `/opt/venv` packages.

## Important scope distinction

The resulting tree fully resolves the original issue's HTTP 500 contract, but
the candidate itself contributes tests/reports only. For included routers, the
existing guard labels metrics with the concrete request path rather than the
parameterized route template. Thus `/v2/items/7` and `/v2/items/8` are distinct
Prometheus endpoint labels. That cardinality behavior is a real remaining
counterexample to a stronger template-preservation claim, but it is not the
crash reported in the original issue.

The original four-node Ascend 910b/NPU GLM workload was not available and was
not run. The deterministic CPU fixture directly exercised the failing
FastAPI/Starlette middleware boundary before model execution; it does not
qualify the original model architecture, NPU backend, or distributed topology.

Raw command output and fetched issue/PR metadata were preserved outside the
checkout at `/job/review-evidence-j-7370141abef7/` while revisions were
switched.
