# Independent review of Jamba 1.5 candidate

Upstream issue: https://github.com/sgl-project/sglang/issues/1190

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/3074

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/3163

Candidate: https://github.com/amdpilot-org/sglang/pull/3159 at `ef1d2fba188d3c59748218908e7986f05c905305`

## Recommendation

Accept the candidate as a substantive, independently verified implementation, while recording it as a partial rather than fully qualified resolution of the original deployment request.

The recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365` has no native Jamba class: its registry resolves `JambaForCausalLM` to the generic Transformers fallback. The exact candidate resolves it to `sglang.srt.models.jamba.JambaForCausalLM` and adds the hybrid attention/Mamba-1 model, recurrent state integration, MoE loading/routing, and focused tests.

All five candidate tests passed on ROCm. Independent adversarial selective-scan checks also passed for FP32/BF16, empty/single/multi-token sequences, varied batch/dimension/state shapes, nonzero initial state, and omitted optional D/z terms.

I did not rely on the candidate's reported server run. I independently generated a deterministic four-layer tiny Jamba fixture containing attention layers, Mamba-1 layers, and two active MoE experts. Transformers and the candidate's real SGLang server ran it on the assigned GPU. `/generate` returned completion IDs `[0, 4, 21, 0, 4, 21, 0, 4]`, exactly matching Transformers. OpenAI completion, batched generation, and streaming probes also returned HTTP 200. The owned server process group was fully cleaned up.

## Scope boundary

This does not fully qualify the original Jamba 1.5 request. Official AI21 Jamba 1.5 repositories returned HTTP 401, and the approximately 52B/399B checkpoints cannot be qualified on the assigned single MI355X. Consequently official-checkpoint loading, pretrained semantic accuracy, tensor parallelism, and deployment-scale memory/performance remain unverified. The portable prefill recurrence is correctness-first and its long-context performance was not measured.

No C++ or HIP source changed, so rebuilding native code was not applicable. Source imports were forced through `/job/repo/python`; the runtime loaded the candidate checkout's native Jamba Python model, while existing accelerator extensions came from the pinned prepared environment.

Raw evidence is retained in `evidence/`; the complete switching-time evidence also remains outside the checkout at `/job/review-evidence-j-6a91e6e55400`.
