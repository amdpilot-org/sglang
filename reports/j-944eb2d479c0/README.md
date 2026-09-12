# Independent review of candidate PR 2283

Reviewed exact commit `08a1faa06f054b939b1a672f3158b90632c713a2` against recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365` and the original queued streaming-session abort contract.

Recommendation: **request changes**. The patch is a real partial fix, not test-only hardening: the base fails the explicit abort and Mamba ordering regression, while the candidate passes it and the adjacent timeout/session suite. It does not fully cover pre-execution queued aborts. A queue-full rejection still returns an abort while leaving the real streaming `Session._inflight` flag set. The priority eviction branch in the same function has the same missing cleanup by inspection.

The candidate modifies only Python source, so no native rebuild applies. `evidence/import-paths.txt` confirms the tested scheduler/session code was imported from this checkout. The GPU attempt used one gfx950 ROCm device and the qualified deterministic tiny Llama. It reached real engine execution and queued-abort HTTP traffic, but the candidate harness hung without writing structured results; the log is retained only as diagnostic evidence.

Raw evidence:

- `evidence/base-regression.txt`: failing-before run on the recorded base.
- `evidence/candidate-focused.txt`: 15 passing focused tests on the exact candidate.
- `evidence/adversarial-queue-limit.txt`: independent remaining counterexample on the exact candidate.
- `evidence/import-paths.txt`: source and runtime import paths plus GPU architecture.
- `evidence/http-candidate-server.log`: incomplete, diagnostic gfx950 server attempt.

Upstream issue: https://github.com/sgl-project/sglang/issues/31765

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/2208

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2318
