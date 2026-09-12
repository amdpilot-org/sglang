# Independent review of PR 749 at `0e62b84`

Recommendation: **accept**. The candidate is a full source-level fix for the original argument-forwarding defect, with CUDA/Blackwell execution explicitly unverified due to the assigned ROCm architecture.

Upstream issue: https://github.com/sgl-project/sglang/issues/38871

Mirror issue: https://github.com/amdpilot-org/sglang/issues/782

Candidate: https://github.com/amdpilot-org/sglang/pull/749 at exact commit `0e62b84dd92e886fb4567887881439a2fa8c3458`

Prepared base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

## Findings

On the prepared base, `VisionAttention.forward()` correctly derived the effective window and tensor-parallel sink slice, but `VisionFlash4Attention.forward()` discarded both before its external `flash_attn_func` call. Running the candidate's regression unchanged on that base produced five failures. Four directly observed the missing `window_size`; the full path also produced output different from an independent local-window/sink attention reference.

At the exact candidate commit, the implementation adds the same forwarding shape already used by the neighboring FA3 path: it always passes `window_size`, conditionally passes `sinks` only when `s_aux is not None`, and continues passing `ver=4`. All five candidate tests passed. An independent adversarial script additionally passed with precomputed metadata, an asymmetric `(0, 2)` window, a zero-valued (but present) sink tensor, `full_attn=True`, and a layer with no sinks.

No source/native ambiguity was found. Runtime inspection resolved `sglang` to `/job/repo/python/sglang/__init__.py` and the reviewed module to `/job/repo/python/sglang/srt/layers/attention/vision.py`. The candidate changes one Python implementation file, one Python test, and prior-job report artifacts; it changes no C/C++/CUDA source, so no native rebuild was applicable.

## Architecture limit

The interpreter reports Torch `2.11.0+rocm7.2`, HIP `7.2`, and an AMD Instinct MI350X. The tests use CPU tensors and an instrumented FA4 boundary. They are strong evidence for the exact dropped-kwargs contract but are not CUDA kernel execution. This environment cannot physically qualify the CUDA-only FA4 implementation, NVIDIA Blackwell behavior, MiMo-V2.5 model execution, or end-to-end multimodal quality.

## Evidence

- `evidence/base-regression.log` and `.xml`: failing-before run, 5 failed.
- `evidence/candidate-regression.log` and `.xml`: exact candidate run, 5 passed.
- `evidence/candidate-adversarial.log`: independent edge cases, passed.
- `evidence/candidate.diff`: exact base-to-candidate diff.
- `evidence/candidate-pr.json`, `upstream-issue.json`, and `mirror-issue.json`: captured review inputs.

Detailed machine-readable conclusions are in `result.json`.
