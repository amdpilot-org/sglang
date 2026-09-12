# Independent review of PR 3257

Reviewed exact candidate `742d70f1088d617a50e85c86f6ba84ade84350a1` against recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`.

Recommendation: **request changes**. The patch is a real partial fix, not test-only hardening: it repairs the concrete response-envelope defect from PR 3237, and the checked-out source passes its focused regression suite. It does not fully resolve the original issue.

## Findings

1. The base reproduction confirms `action_generation_response()` drops `candidate_group` and `candidates` even when the internal output contains them. The candidate retains both fields; JSON candidate arrays become lists and the NumPy/msgpack-oriented path retains arrays.
2. Candidate source was imported from `/job/repo/python/sglang/...`, not an installed stale copy. No native source changed, so a native rebuild was not applicable.
3. The candidate's focused suite passed: 157 tests and 43 subtests.
4. An independent MI350X reducer check matched a float64 accumulation/division reference with zero observed absolute error for the test tensor.
5. Remaining concrete counterexample: for `Pi05SamplingParams`, `parameters.candidate_trajectory={"count":2,"reducer":"mean"}` is silently ignored and `num_outputs_per_prompt` remains 1. The original proposal explicitly requires pipelines without an action-candidate adapter to reject the field. This silent downgrade can mislead a client into believing candidate ensembling was requested.

## Scope limitations

Cosmos3 weights were unavailable. Model-backed action serving, full denoising, candidate denormalization, batched/sequential model parity, CFG variants, TP/SP, cancellation/failure behavior, dynamic co-batching, semantic quality, and multi-topology benchmarks remain unverified. The candidate also does not implement the issue's requested candidate observability metrics. The qualified tiny Llama fixture cannot validate a Cosmos3 action architecture and was not treated as proof.

Raw commands and outputs are retained beside this report.
