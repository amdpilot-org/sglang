# Independent review of candidate PR 2349

Upstream issue: https://github.com/sgl-project/sglang/issues/31205

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/2279

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2385

Candidate: https://github.com/amdpilot-org/sglang/pull/2349 at
`1d3dec7eab68bfaf1e28ab1068b2438543d3c406`.

## Recommendation: request changes

The candidate is useful test-only hardening for the exact reported geometry, but
it overstates the prepared source as a complete correction for intrinsically
oversized SWA admissions. The source makes progress for the reported 20,992 SWA
tokens / 32,768 chunk / 512 page / 20,742 prompt case by admitting a 19,968-token
chunk. It does not cover every request that can never fit the total pool.

### Blocking finding

`PrefillAdder.add_one_req` detects that a request can never fit using
`_swa_req_never_fits`, then tries `_swa_chunk_cap`. If the fully drained pool is
no larger than the reserved page/decode headroom, `_swa_chunk_cap` is zero and
the method still returns `NO_TOKEN`. Repeating admission does not change that
outcome, so an FCFS queue head can still wait forever and starve runnable work
behind it. An independent case with `size_swa=rem_swa=512`, `page_size=512`,
`max_new_tokens=1`, and a 2,000-token prompt reproduced three consecutive
`NO_TOKEN` results with no extend range. The candidate does not test or disclose
this branch.

This is tied directly to the issue's contract: a request whose budget exceeds
the *total* SWA capacity must not be silently deferred forever. Safe chunking is
a valid alternative to HTTP 400 when a chunk fits, but a zero-cap case still
needs fail-fast rejection or another terminal outcome.

## What was verified

- The candidate modifies only tests and reports; production
  `schedule_policy.py` is byte-identical to the recorded base.
- Imports resolve to `/job/repo/python/sglang/srt/managers/schedule_policy.py`;
  Torch resolves to the prepared ROCm 7.2 environment.
- The candidate's focused tests pass (7 passed), and the complete PrefillAdder
  module passes (30 tests and 12 subtests).
- Independently simulating the old gate at the reported geometry repeatedly
  returns `NO_TOKEN` for the long head, while a 128-token tail is independently
  runnable. The current source admits the reported prompt and prompts through
  65,536 tokens as bounded 19,968-token first chunks.
- The prepared base already contains the production escape hatch. Therefore an
  actual failing-before run on that revision is impossible; the candidate's
  failing-before evidence is a monkeypatch simulation, not execution of the
  historical buggy implementation.

## Environment limits

One visible AMD Instinct MI350X (`gfx950`) was available. These scheduler unit
tests execute no GPU kernels, so `gpu_execution` is false and no numerical GPU
claim is made. The candidate has no C++/HIP/native changes, so no native rebuild
was applicable. DeepSeek-V4-Pro/DSpark weights and the reported 1P1D TP4+TP4,
mooncake, multi-node, 8xB300 deployment were unavailable; no full HTTP serving,
model-semantic, transfer, or distributed reproduction is claimed.
