# Final-prefill abort commit race

Upstream issue: https://github.com/sgl-project/sglang/issues/34149

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1659

The bug was present at the prepared base `358c163250ad3b1f62939b01ce1314a0a31a0365`.
With only the candidate regression tests applied, the focused run produced three
failures: the pending-finish prefill request was admitted to optimistic decode,
the delayed token `101` was appended to the aborted request, and a sampling-mask
finish replaced the earlier abort. The complete output is retained in
`raw/failing_before.log`.

The correction protects both state-commit boundaries. While an overlap result is
pending, a pending-finish prefill request is excluded from the prefill-to-decode
merge (without excluding decode members of a mixed batch). When the delayed
prefill result is consumed, the pending finish is promoted with zero newly
accepted tokens. Token-derived hidden state, custom metadata, logprobs, sampling
mask, and grammar state are not attached, while packed cursors still advance for
later live requests and normal finished-request cleanup releases resources.

After the source change, the focused suite passed 20 tests and 2 subtests. Extra
deferred-abort and grammar boundaries passed 8 tests. Python compilation and
`git diff --check` also passed. See `raw/passing_after.log` and
`raw/additional_boundaries.log`.

A qualified deterministic tiny random Llama fixture was generated outside the
worktree using the scripts from amdpilot-org/sglang PR 649 at commit
`f1d603677ca76a9ea21124a544e405c5b0cbd315`. The patched source checkout ran on
the assigned AMD Instinct MI355X (`gfx950`) and returned HTTP 200 for ordinary
generation, OpenAI completion, a two-request batch, and an eight-event streaming
completion. Raw requests/responses and the server log are under
`raw/gpu_server/`. The server log records actual GPU prefill/decode execution and
Triton compilation for gfx950. GPU utilization and allocated VRAM returned to
zero after cleanup.

The GPU run qualifies transport and engine execution only. Its synthetic random
weights do not validate model semantics, and the HTTP probe did not control the
scheduler at the exact delayed-final-prefill abort instant. The deterministic
race reproduction and boundary assertions are therefore the focused scheduler
regressions, not the startup or HTTP smoke. No multi-GPU, PP, DP, distributed, or
Qwen runtime claim is made.
