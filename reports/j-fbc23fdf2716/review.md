# Independent review of candidate f06c14a

Upstream issue: https://github.com/sgl-project/sglang/issues/33415

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/3399

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/3424

Candidate PR: https://github.com/amdpilot-org/sglang/pull/3421

## Verdict

Recommendation: **accept**. The candidate fully resolves the original source-level dispatch defect. It changes the non-capture branch of `Olmo2Attention._apply_qk_norm` from a direct `forward_native` call to normal `RMSNorm` module dispatch while preserving shapes and the existing capture-only alternate-stream behavior. No counterexample was found.

This is a full fix for the reported branch-selection defect, not merely test hardening. The original issue is performance-only; both before and after outputs were numerically correct. End-to-end OLMo-2 and H200 performance claims remain unverified here because weights and NVIDIA hardware were unavailable.

## Evidence

The prepared branch was exactly the recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`. Candidate `f06c14a3339a03ed7b89f0a8737344fee0c7c080` has that commit as its sole parent. Imports resolved to `/job/repo/python/sglang/...`, and Torch reported `2.11.0+rocm7.2`, HIP `7.2.26015`, and one AMD Instinct MI350X (`gfx950`).

On the base, an independent harness invoked the actual `Olmo2Attention._apply_qk_norm` implementation with actual `RMSNorm` modules. Across BF16, FP16, and FP32; 1, 7, and 129 tokens; and packed-QKV noncontiguous views, the eager branch entered `forward_native` twice and did not enter public ROCm dispatch. All outputs matched an independent FP64 RMSNorm formula, confirming the reported defect is dispatch/performance rather than correctness.

At the exact candidate, its two focused regression tests passed. The same independent harness showed both norms entered normal backend dispatch. In the prepared default environment, `forward_hip` then fell back to native because vLLM RMSNorm was unavailable and AITER was not enabled; this is backend policy, not the OLMo-2 bypass. With `SGLANG_USE_AITER=1`, each norm selected `forward_aiter`, built/loaded the real AITER RMSNorm module from the private runtime cache, avoided `forward_native`, and matched the independent FP64 reference for every case. Maximum observed absolute errors were 0.0078125 (BF16), 0.001953125 (FP16), and 4.77e-7 (FP32).

An independent real ROCm graph captured the existing alternate-stream branch with Q width 2048 and K width 1024. Replay after replacing the static inputs matched the independent reference exactly and used `forward_aiter`. Thus the candidate preserves the already-correct capture/replay path while fixing eager execution used by prefill and decode with graphs disabled.

No SGLang native, FlyDSL C++, or build-system source changes are present. A repository native rebuild was therefore not applicable. The AITER RMSNorm extension used by the independent kernel validation was built in `/tmp/amdpilot-repo-j-fbc23fdf2716/cache/aiter/` and loaded by the tested process.

## Limitations

- OLMo-2 weights were unavailable, so full-model serving, semantic accuracy, and end-to-end prefill/decode timing were not run. The tiny Llama fixture was not substituted because it cannot validate this architecture.
- No NVIDIA H200 was assigned. The upstream H200 timing remains source evidence, not a measurement from this review.
- Tensor-parallel execution was not run, although the unchanged gather/norm/split ordering was inspected.
- The unrelated CUDA FA3 illegal-address failure was kept separate and cannot be reproduced on this AMD-only host.
- AMD timing was not used to validate the upstream H200 speedup claim. The review validates dispatch and real-kernel numerical behavior.
