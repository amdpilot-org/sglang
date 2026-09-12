# Independent review of PR 2146

- Upstream issue: https://github.com/sgl-project/sglang/issues/33088
- Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2180
- Candidate: https://github.com/amdpilot-org/sglang/pull/2146
- Exact candidate commit: `286168e74709454e5db1515306932ed21995212d`
- Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

## Recommendation

**Accept as a bounded mitigation, not as a fully demonstrated resolution of the original distributed serving failure.**

The candidate makes a substantive source change. For each decode queue and bootstrap address it replaces repeated scheduler-cycle queries with exponential backoff (100 ms, doubling to a 2 s ceiling), caps an unresolved cohort at eight attempts, aborts its receivers, and removes those requests from the pending lookup queue. It also preserves successful recovery when the mapping appears on the eighth attempt. This directly fixes the prior candidate's unbounded pending request and fixed-interval retry behavior.

The change does not implement once-per-node ownership or cross-process caching. Eight independent DP queues each permit an initial attempt and can perform up to 64 attempts for one unresolved cohort. A later cohort receives a fresh retry budget. This is bounded and dramatically below the reported 1.36 million-call storm, but it remains process-local fan-out.

## Failing-before evidence

I temporarily installed only the candidate's three retry-specific tests while retaining the recorded base implementation, then ran:

```bash
PYTHONPATH=/job/repo/python /tmp/amdpilot-repo-j-7493f317b56c/venv/bin/python \
  -m pytest -q test/registered/unit/disaggregation/test_decode_queue_cleanup.py \
  -k 'unresolved_prefill_dp_rank_queries_are_rate_limited or dp_rank_query_throttle_is_independent_per_bootstrap or unreachable_dp_rank_query_retries_are_bounded'
```

All three failed. Most importantly, ten scheduler resolutions produced ten bootstrap calls rather than stopping at three, and three rapid prefetch cycles submitted three calls rather than two. Raw output is in `raw/base-counterexamples.txt` (exit 1). The test file was restored before switching revisions.

## Candidate and independent evidence

At exact candidate commit `286168e74709454e5db1515306932ed21995212d`:

- The focused candidate file passed: 12 tests.
- The complete registered disaggregation unit directory passed: 361 tests and 45 subtests.
- An independent harness verified an eight-call terminal bound, pending-queue removal, one receiver abort, and successful initialization when the mapping becomes available on the eighth attempt.
- The same harness instantiated eight queues and observed eight initial calls, proving throttle state is not shared once per node.
- Two successive permanently failing cohorts with a two-attempt limit made four calls total, proving terminal state is per cohort rather than a permanent circuit breaker.
- Import inspection resolved `sglang.srt.disaggregation.decode` to `/job/repo/python/sglang/srt/disaggregation/decode.py` with `PYTHONPATH=/job/repo/python`, not to an installed SGLang copy.

`git diff --check` found trailing whitespace in three candidate report text files. This does not affect the runtime correction, but is retained in `raw/candidate-diff-check.txt`.

## Classification and limitations

This is a **partial original-issue fix**, not test-only hardening and not an unverified source claim. The bounded retry/backoff contract is reproduced failing on the base and passing on the candidate. The exact reported outcome—elimination of `[Errno 99]` and successful Qwen3.5-397B-A17B-FP8 service in a two-node TP8/EP8/DP8 MoRI deployment—remains unverified.

The environment provides one AMD Instinct MI355X (`gfx950`) GPU. It does not provide a second node, eight GPUs per role, the stated model weights, or the reported MoRI topology. No GPU execution is relevant to the CPU-side queue tests, and no model-serving smoke is presented as proof. No C++ or other native source changed, so a native rebuild was not applicable. The existing AITER import resolved to `/tmp/amdpilot-repo-j-7493f317b56c/cache/aiter/module_aiter_core.so`.
