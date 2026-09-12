# Independent review of candidate PR 3377

Reviewed exact candidate commit `01616c1d61bed6ff08ef59f8e8b981b0439086ba`
against recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`.

Upstream issue: https://github.com/sgl-project/sglang/issues/31954

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/3338

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/3380

## Recommendation

Accept. The base implementation repeatedly displaced a cache-cold request in a
two-request shallow queue for every newly arriving 100-token-match request. The
exact candidate's opt-in aging scheduled that cold request after four
displacements with `25` aging tokens per pass. Independent non-divisible cases
matched the claimed `ceil(match / aging)` displacement bound. Disabled mode,
temporary deprioritization, the 128/129 fallback boundary, CLI transport, and
negative validation were also checked.

The candidate changes Python configuration and scheduler state only. No native
source changed, so no native rebuild was applicable. Both `sglang` and
`schedule_policy.py` imported from `/job/repo/python` while each revision was
checked out.

## Limitations

This review validates deterministic scheduler ordering, not the production
latency distribution. The reported GLM-5.2-NVFP4 weights, 8xB300 topology,
~115K-token workload, and production cache-hit distribution were unavailable.
The assigned single gfx950 GPU was not used because the defect and candidate
are confined to Python queue ordering; a tiny Llama transport fixture would not
qualify the reported architecture, scale, or latency behavior.

Commands and outputs are summarized in `raw/review-evidence.txt`; structured
claims are in `result.json`.
