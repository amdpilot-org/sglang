# Independent review of PR 2472

Reviewed `https://github.com/amdpilot-org/sglang/pull/2472` at exact commit
`f4a3d6086303839480b83005c4a690705c826604` against upstream issue
`https://github.com/sgl-project/sglang/issues/31765` and mirror issue
`https://github.com/amdpilot-org/sglang/issues/2506`.

Recommendation: **accept**. The candidate fully resolves the original queued
streaming-session lifecycle defect at the scheduler boundary and also resolves
the two queue-limit counterexamples inherited from the review of PR 2283.

On the recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`, the
candidate regression produced four failures and one pass: explicit queued
abort did not clear the streaming session, waiting-timeout/Mamba cleanup lacked
`FINISH_ABORT`, incoming queue-limit rejection did not clear the session, and
priority eviction lacked both lifecycle cleanup and the Mamba abort marker.
At the exact candidate commit all five regression cases passed. Adding the
existing streaming-session cache suite produced 14 passes total.

Source inspection confirms the candidate centralizes pre-execution streaming
cleanup in `_abort_queued_streaming_session`: it installs `FINISH_ABORT` before
Mamba cache release and calls `Session.abort_req()` without modifying the last
committed request nodes. Both queue-limit outcomes and the ordinary queued
abort/timeout path call this helper. Non-streaming requests remain untouched.

The imported modules came from `/job/repo/python/sglang`, including
`/job/repo/python/sglang/srt/managers/scheduler.py`. The change is Python-only;
no native source changed and no native rebuild was applicable.

The assigned GPU is an AMD Instinct MI350X (`gfx950`) with Torch
`2.11.0+rocm7.2` / HIP 7.2. A source-checkout server loaded and executed the
qualified deterministic tiny Llama fixture, including a completed streaming
session first turn. The attempted concurrent HTTP queue probe did not establish
the intended queue ordering and is retained only as inconclusive diagnostic
evidence, not as proof. No Qwen or hybrid-Mamba weights were available, so the
exact reporter model and a real Mamba end-to-end run remain unverified. The
deterministic scheduler/cache tests directly cover the pre-model-step contract.

Raw outputs are under `evidence/`.
