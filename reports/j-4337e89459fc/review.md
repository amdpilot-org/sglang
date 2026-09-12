# Independent review of candidate 970ab777f74aea05733c8f5444642f26006c3f59

Upstream issue: https://github.com/sgl-project/sglang/issues/36481

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1274

Candidate: https://github.com/amdpilot-org/sglang/pull/1237 at exact commit `970ab777f74aea05733c8f5444642f26006c3f59`.

## Finding

Recommendation: **request changes**. The candidate is a useful crash-avoidance mitigation, but it does not fully resolve the original issue's compact target-verify CUDA-graph contract.

The change checks `supports_ragged_verify_graph` before `DecodeCudaGraphRunner.capture()`. For the reported hybrid backend, `HybridLinearAttnBackend.supports_ragged_verify_graph` is false because `TritonAttnBackend` retains the base default of false. The new early return therefore prevents entry into the failing capture/warmup path. At replay admission, the existing `_can_run_ragged_verify_graph()` check also returns false, causing target verification to execute eagerly.

Consequently, the candidate does not repair `extend_attention_fwd`, enable Triton ragged graph metadata, or capture/replay a compact target-verify graph. It makes the affected `compact` configuration fall back to eager target verification. This avoids the boot crash, but forfeits the graph path whose compact-mode production behavior the report is about. The issue already documents `static` and `cap-accept` as usable workarounds; this is another automatic fallback rather than verification of compact graph support.

## Evidence

- Recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365` exactly matched the image-prepared checkout.
- Running the candidate's complete six-test file against the base produced one error: `test_compact_capture_skips_unsupported_backend` entered `warmup()` and raised `AssertionError: must not warm up`. The other five tests passed. Raw log: `/job/review-evidence-j-4337e89459fc/base_candidate_test.log`.
- At the exact candidate commit, the same file passed all six tests. Raw log: `/job/review-evidence-j-4337e89459fc/candidate_regression.log`.
- Source imports were confirmed from `/job/repo/python/sglang`, including `decode_cuda_graph_runner.py`; the inspected `capture()` contained the candidate guard.
- Independent source-path analysis confirmed the existing replay admission guard returns false for unsupported ragged backends, and `ModelRunner._forward_raw` then uses the eager forward path.
- `git diff --check` passed.

## Limitations and remaining counterexamples

The assigned device was one AMD Instinct MI355X (`gfx950`) under ROCm 7.2 with Torch `2.11.0+rocm7.2`. The reporter used an NVIDIA B300 with CUDA 13.2. Qwen3.5/Qwen3.6 and DSpark weights were unavailable. Therefore the original illegal-memory-access failure in the Triton CUDA kernel, the exact `bs=20`/8-tokens-per-request graph shape, full-model startup, and serving behavior could not be reproduced here.

No native source changed, so no native rebuild was applicable. No GPU numerical claim is made: the candidate's regression is a mocked Python control-flow test and does not execute the reported attention kernel.

Remaining counterexample: on the exact affected backend, `SGLANG_RAGGED_VERIFY_MODE=compact` still cannot run target verification through CUDA graphs; it is admitted only to eager execution. A full fix needs either safe compact graph support for the Triton/hybrid path, with the reported shape validated on the relevant NVIDIA architecture, or an explicit product decision and user-facing contract that this backend is unsupported and will fall back.
