# Compact target-verify correction generation 2

Upstream issue: https://github.com/sgl-project/sglang/issues/36481

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1522

Candidate PR: https://github.com/amdpilot-org/sglang/pull/1380 at exact commit `62688251058491fc6bb5251a95bc187c3019dea8`

Independent review PR: https://github.com/amdpilot-org/sglang/pull/1485

## Result

The candidate is a valid crash-avoidance fallback and is preserved unchanged. For compact target verification, `DecodeCudaGraphRunner.capture()` now skips capture when the selected attention backend does not advertise `supports_ragged_verify_graph`; replay admission remains false and verification uses eager execution. Supported compact backends and non-compact capture continue into warmup/capture.

The review's remaining counterexamples are confirmed:

- `TritonAttnBackend.supports_ragged_verify_graph` is false. A hybrid full-attention/GatedDeltaNet backend requires both child backends to support ragged verify graphs, so a Triton-backed Qwen hybrid does not use a compact target-verify CUDA graph after this change.
- The available GPU is AMD Instinct MI355X/gfx950 with ROCm 7.2, not NVIDIA B300 with CUDA 13.2.
- The Qwen hybrid and DSpark model weights used by the report are unavailable. The regression exercises capture/replay admission with mocks; it does not execute `extend_attention_fwd` or a hybrid model.

Consequently, no source change to Triton attention is justified by this environment. In particular, the reported illegal memory access at batch size 20 and eight tokens per request remains unverified. The tiny Llama serving fixture is not applicable because it cannot qualify the different hybrid architecture or the failing kernel.

## Evidence

On the exact candidate, the six capability/admission tests pass (`evidence/candidate-exact.log`). Removing only the candidate's guard while retaining its test makes `test_compact_capture_skips_unsupported_backend_and_replay` fail because `capture()` enters the poisoned `warmup()` (`evidence/failing-before.log`). Restoring the guard passes all six tests again (`evidence/passing-after.log`). Runtime and backend capability facts are in `evidence/environment-and-capability.log`.
