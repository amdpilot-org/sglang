# Independent review of candidate PR 587

Recommendation: accept. The candidate at `626176a4a99710f1323c7f236fd8d64f19bedf44` fully resolves the original reported admission-path failure.

On the prepared base `358c163250ad3b1f62939b01ce1314a0a31a0365`, a 1024-token admitted range inside a 1500-token encoder reproduces the original bookkeeping corruption and fails with `AssertionError: Expected 0, got -476`. The candidate changes the actual `PrefillAdder` path so an uncached encoder is never split: a later request is deferred when the current chunk is partly consumed, while an oversized encoder is admitted through its complete boundary in an otherwise-empty batch.

The exact candidate's focused tests passed, including page sizes 1 and 16, deterministic alignment 128, chunk sizes below/equal/above the encoder boundary, deferral under concurrency-like consumed budget, downstream cache-location metadata, and preservation of decoder-only multimodal behavior. The complete two-file focused suite passed with 36 tests and 28 subtests.

No native code changed, so no native rebuild was applicable. Imports resolved to `/job/repo/python/sglang` and the checked-out `schedule_policy.py`. The environment had PyTorch 2.11.0+rocm7.2 and one AMD Instinct MI350X. Intel XPU and a full Whisper server/model run were unavailable and therefore not claimed. Raw evidence is retained under `/job/raw/j-8ed465575f72/`.

Upstream issue: https://github.com/sgl-project/sglang/issues/38627

Mirror issue: https://github.com/amdpilot-org/sglang/issues/589
