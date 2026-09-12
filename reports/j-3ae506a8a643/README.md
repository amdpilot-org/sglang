# Independent review of PR 3381

Reviewed candidate commit `095558b5a591b46e509d992f4645c923faf2c1e2`
against recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`.

Recommendation: **accept**. The candidate resets the previous scheduler pass's
`batch_is_full` admission hint before the early return, allowing current slot
capacity to be evaluated. On the base, the candidate's three regressions all
failed and an independent stale-hint test failed. At the exact candidate, the
full focused test file passed (6 tests), and three independent tests passed:
stale capacity is reconsidered, genuine zero capacity re-latches fullness
without admission, and an active chunk continuation still bypasses the normal
slot gate.

The candidate changes Python only. No C++, FlyDSL, or other native source was
changed, so a native rebuild was not applicable. The prepared interpreter
imported `scheduler.py` from `/job/repo/python/sglang/srt/managers/scheduler.py`.

The available accelerator was one AMD Instinct MI355X (`gfx950`) with ROCm 7.2.
This review did not rerun a server because the candidate already retained a
qualified tiny-Llama gfx950 serving trace, and the independent source-level
tests directly exercise the faulty gate and its safety boundaries. That trace
shows a transient waiting request admitted before existing requests finished,
but it validates only the generic Llama transport/engine path. The reported
Qwen3.8-27B-NVFP4 hybrid GDN/mamba model, NEXTN speculation, FP8 KV cache,
Docker/WSL2 setup, semantic accuracy, and multi-minute workload were not
available and were not reproduced.

No functional counterexample was found. A documentation/evidence inconsistency
does not affect the fix: the candidate says `git diff --check` passed, but a
fresh check reports trailing whitespace in captured pytest log artifacts.

Upstream issue: https://github.com/sgl-project/sglang/issues/35537

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/3335

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/3388

Candidate PR: https://github.com/amdpilot-org/sglang/pull/3381

