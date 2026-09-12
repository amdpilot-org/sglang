# Correction generation 2: scheduler fatal-state hardening

Candidate: https://github.com/amdpilot-org/sglang/pull/997 at exact commit
`7035931a1eee025b1847b58c531c583f983ea8a6`.

Independent review: https://github.com/amdpilot-org/sglang/pull/1850.

Upstream issue: https://github.com/sgl-project/sglang/issues/38167

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1859

## Result

The candidate's valid monolithic scheduler fix is retained. Two independently
reproduced gaps are corrected:

- scheduler RPCs now have a finite one-hour default receive deadline instead
  of leaving ZeroMQ `RCVTIMEO=-1`;
- a disaggregated encoder latches the same fatal accelerator state, returns an
  error for the triggering work item, and rejects later work without another
  worker dispatch.

Ordinary allocation failures remain recoverable. No resolution-based admission
rule or conversion of `device not ready` to an out-of-memory diagnosis was
added because the reported architecture, CUDA stack, model weights, and LoRA
were unavailable and the causal claim could not be verified.

## Evidence

At the exact candidate commit, the probe recorded in `raw/candidate_before.txt`
measured `effective_timeout=None`, actual ZeroMQ `RCVTIMEO=-1`, two encoder
dispatches after the first fatal signature, and no fatal state.

After the correction, the focused suite in `raw/focused_after.txt` passed 216
tests and 62 subtests. Its regressions assert the 3,600,000 ms socket timeout,
the fatal error reply for the triggering disaggregated request, and rejection
of the next request without worker dispatch.

## Limitations

The prepared host provides ROCm 7.2 on gfx950, not RTX 5070 / WSL2 / CUDA 13.0.
MiniMax-H3 weights and the reporter's LoRA were unavailable. The exact
1344x768 failure, whether it is memory exhaustion, and the subsequent real
480p request remain unverified. No GPU workload or native rebuild was needed
for these deterministic control-flow regressions.
