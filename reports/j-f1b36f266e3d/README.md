# Streaming-session disconnect ownership fix

Upstream issue: https://github.com/sgl-project/sglang/issues/36475

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1174

The issue is a composite race. A disconnected request can remain in flight
while an immediate retry is rejected. The rejected request never acquired the
session-wide inflight slot, but `StreamingSession.find_active_slot` previously
called `Session.abort_req()` unconditionally and released the slot owned by the
disconnected request. A subsequent request could then be admitted concurrently
and mutate the owner's shared token arrays.

The fix records inflight ownership on each admitted `Req`. Pre-abort cleanup
only releases the session slot when that request owns it. Successful completion
and mid-processing abort both clear the ownership marker.

## Evidence

- `evidence/unit-before.log`: the two issue-specific regressions fail against
  the original source. The rejected overlap clears `_inflight`, and a third
  request is incorrectly admitted.
- `evidence/unit-after.log`: 16 focused session/cache tests pass, including the
  two regressions and the independent boundary that an admitted request still
  releases its own slot on abort.
- `evidence/http-before/`: real gfx950 server execution reproduced the immediate
  retry returning `prompt_tokens=1`, `cached_tokens=0`, after the disconnected
  request started with the expected inherited context.
- `evidence/http-after-final/`: real gfx950 server execution followed the clean
  rollback branch: the immediate retry completed with the expected 72 prompt
  tokens and `/health` remained HTTP 200.
- `evidence/gpu.json`: assigned GPU, architecture, Torch, and HIP identity.

The HTTP fixture is the qualified deterministic random tiny Llama from mirror
PR 649 at commit `f1d603677ca76a9ea21124a544e405c5b0cbd315`. Its weights were
generated under `/tmp/amdpilot-repo-j-f1b36f266e3d`, outside the worktree. It
validates transport, scheduler execution, session token accounting, and GPU KV
lifecycle only. Qwen2.5 weights were unavailable, so this does not validate
Qwen semantics. The small fixture reproduced context loss but not the reported
idle invariant crash.

## Reproduce

```bash
PY=/tmp/amdpilot-repo-j-f1b36f266e3d/venv/bin/python
$PY -m pytest -q \
  test/registered/unit/mem_cache/test_session_token_share_unit.py \
  test/registered/unit/mem_cache/test_streaming_session_unit.py
$PY reports/j-f1b36f266e3d/run_repro_server.py \
  --fixture /tmp/amdpilot-repo-j-f1b36f266e3d/tiny-random-llama \
  --output /tmp/amdpilot-repo-j-f1b36f266e3d/http-repro
```
