# Independent review of candidate PR 604

Upstream issue: https://github.com/sgl-project/sglang/issues/38924

Mirror issue: https://github.com/amdpilot-org/sglang/issues/607

Candidate: https://github.com/amdpilot-org/sglang/pull/604 at `202c851845976d0c76fccb8e3707a219625f8100`

Recommendation: **accept**. The candidate fully resolves the concrete parser contract described by the original issue for the reproduced malformed shapes.

## Evidence

The prepared base `358c163250ad3b1f62939b01ce1314a0a31a0365` was tested first. The candidate's focused wrapped-argument tests, copied outside the checkout, failed on the base for all original failure classes: nested Direct-JSON wrappers, scalar Direct-JSON wrappers, XML `string="true"` wrappers, and streaming accumulation. The failure exposed `{"arguments": ...}` / `{"input": ...}` instead of the declared tool properties.

The worktree was then detached at the exact candidate SHA. Imports were confirmed to resolve to `/job/repo/python/sglang/srt/function_call/deepseekv32_detector.py` and `/job/repo/python/sglang/srt/function_call/deepseekv4_detector.py`, and the new normalization method was present in the loaded class. The candidate changes Python only; no native source or FlyDSL compiler changed, so no native rebuild was applicable.

At the candidate SHA:

- The complete candidate detector test file passed: 10 tests.
- The broader function-call parser suites passed: 253 tests and 4 subtests.
- Eight independent adversarial tests passed for arbitrary boundaries, bytewise XML chunks, double-encoded dictionaries, Unicode scalar values, mixed parallel calls, legitimate schema fields named `input`/`arguments`, ambiguous scalar schemas, and unchanged well-formed calls.

The worktree was returned to `amdpilot/j-308f0c5b858b` at the recorded base before this report was added. No candidate source was copied or modified on the delivery branch.

## Architecture and limitations

The interpreter was `/tmp/amdpilot-repo-j-308f0c5b858b/venv/bin/python`; SGLang loaded from the prepared checkout. The host exposes an AMD Instinct MI350X with PyTorch 2.11.0+rocm7.2 and HIP 7.2. A small float64 GPU polynomial check exactly matched an independently computed CPU reference (maximum absolute error 0.0), but that check is only environment evidence: DSML parsing is CPU-side and requires no GPU.

A live DeepSeek-V4-Flash-Vision-Exp model/server was not available, so stochastic long-context generation and the OpenAI HTTP layer were not replayed. This does not leave a parser counterexample: the exact malformed payloads supplied by that generation path were reproduced directly against the actual implementation, and both one-shot and streaming outputs were verified against tool schemas.

Raw logs and the independent test source were retained under `/job/review-evidence-j-308f0c5b858b/` during revision switching.
