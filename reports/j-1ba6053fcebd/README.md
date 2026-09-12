# Independent review of PR 1861

Reviewed candidate: https://github.com/amdpilot-org/sglang/pull/1861 at exact commit `6bcff2549ebf675043710280d4ad4702ff77ef67`.

Upstream issue: https://github.com/sgl-project/sglang/issues/38118

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1864

Recommendation: accept. The candidate is a source fix, not test-only hardening. It reproduces the recorded-base failure and corrects both the original held chunk-row double count and the independently reported heterogeneous beam-width counterexample. No remaining scheduler counterexample was found in focused or independent boundary tests.

The full original deployment was not available: this host has one AMD Instinct MI350X gfx950 with ROCm 7.2, rather than eight NVIDIA B200 GPUs with CUDA/TRT-LLM, and the GLM-5-FP8 weights were unavailable. Accordingly, the report treats the deterministic scheduler tests as evidence for the admission contract and does not claim a full-model, distributed, or backend reproduction.

Raw review output was retained outside the revision-switched checkout at `/tmp/j-1ba6053fcebd-review-evidence`. Selected logs are included beside this report.
