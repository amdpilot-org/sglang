# Independent review of amdpilot-org/sglang PR 1515

Candidate: https://github.com/amdpilot-org/sglang/pull/1515
Exact commit: `b1710c2e6d194ce85896c27a5e9033c790bb73cd`
Upstream issue: https://github.com/sgl-project/sglang/issues/34969
Mirror issue: https://github.com/amdpilot-org/sglang/issues/1552

## Verdict

Recommendation: **accept**. The candidate fully resolves the original issue's logical-anchor division-by-zero contract in the available environment.

The recorded base commit was also the prepared checkout commit and is the candidate's direct parent. On that base, an actual `LogicalHostPool` produced `bytes_per_page=0`, and constructing `HiCacheHF3FS` failed at the exact reported division with `ZeroDivisionError`.

At the candidate commit, repository source imports were confirmed for `sglang`, `backend_factory.py`, and `storage_hf3fs.py`. The patch contains Python and tests only, so no native library changed and no native rebuild was applicable.

The submitted four-case regression passed. An independent test went beyond its v1 marker round trip: it registered a physical named v2 side pool, wrote two logical anchor markers, wrote only one physical component and confirmed the usable hit prefix truncated to one page, then wrote the second component and restored both pages' bytes exactly. This verifies that the candidate preserves the DSV4 design in which the primary key is an existence gate while physical data lives in named side pools.

## Evidence

- Base reproduction: exit 1 with `LogicalHostPool.kv_buffer=None`, `bytes_per_page=0`, and the reported `file_size // bytes_per_page` traceback.
- Candidate regression: 4 passed.
- Candidate regression plus HF3FS utility and independent adversarial tests: 7 passed in 96.09 seconds.
- Candidate diff check: passed.
- Candidate commit is a direct child of the required base `358c163250ad3b1f62939b01ce1314a0a31a0365`.

Complete raw command output and the independent test source are preserved outside the revision-switching checkout at `/job/review-evidence/`. Structured claims are in `result.json`.

## Limits and residual observation

This review did not run DeepSeek-V4 weights, TP=2, NVIDIA Blackwell, or a deployed HF3FS cluster. The host provides one AMD Instinct MI350X (`gfx950`) under ROCm 7.2, and `hf3fs_fuse` is unavailable. The deterministic mock validates storage geometry, metadata, file transport, gating, and byte restoration, not full-model or distributed behavior.

An independent boundary check found that a named v2 physical pool reporting zero bytes per page still reaches a `ZeroDivisionError` during registration rather than the candidate's new diagnostic `ValueError`. That malformed side-pool geometry is outside the original DSV4 failure (whose physical side pools have non-zero pages), so it is recorded as a residual hardening opportunity rather than a counterexample to the original fix.
