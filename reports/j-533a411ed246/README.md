# Independent review of PR 993

Reviewed exact candidate commit `f32316b1869fe8fb2207286e6db22d7264410c11` against recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365` and upstream issue https://github.com/sgl-project/sglang/issues/37585.

Recommendation: **accept as a correct partial fix**. The candidate fixes the shared eager hybrid fixed-width verification contract and safely handles fabricated dummy requests in the affected Triton convolution, recurrent GDN, and opt-in fused KDA paths. It does not fully resolve the original issue because GLM DSA KPool has a separate real-request planning boundary addressed by upstream PR 37588.

The prepared base reproduced the issue-specific layout failures: `B=38,D=6,A=8` padded 228 tokens to 232 rather than a complete 240-token request-group boundary, and `B=33` padded 198 to 200 rather than 216. The candidate passed its CPU regression, all 17 assigned-gfx950 numerical/state tests, and 51 independent adversarial layout cases spanning other TP sizes, fixed widths, batch boundaries, zero-token ranks, and DP token domains.

The interpreter loaded SGLang source from `/job/repo/python/sglang`. No C/C++/HIP/FlyDSL native source changed, so no native rebuild was applicable. The changed Python/Triton modules compiled successfully.

The reported Qwen3.5-35B-A3B weights, 8x NVIDIA H20 TP8/EP8 environment, CUDA 13 runtime, and GLM-5.3-Flash weights were unavailable. Therefore this review does not claim an end-to-end HTTP/model reproduction, NVIDIA/CUDA qualification, multi-rank collective execution, semantic accuracy, or validation of the GLM DSA KPool boundary.

Raw logs and the reviewed source diff are retained in `raw/`.
