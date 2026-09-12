# Independent review of EAGLE radix-cache candidate

Upstream issue: https://github.com/sgl-project/sglang/issues/32459

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/2073

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2170

Candidate PR: https://github.com/amdpilot-org/sglang/pull/2137

Reviewed exact commit `42762d9410295da6b67c93bd1a734bfd7cd12811`
against recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`.

## Verdict

Recommendation: **accept as a source-level partial fix**. The candidate removes
the EAGLE-only two-step KV reservation from both the capacity check and actual
allocation, bases both on the synchronous target sequence length, preserves the
double reserve for UNO, and handles attention backends that clear
`seq_lens_cpu`. Its regression fails on the recorded base (3 failed, 1 passed)
and passes at the candidate (5 passed including the UNO compatibility test).
Independent adversarial checks also passed for selected-request indexing,
page rounding, stale committed lengths, the missing-CPU-mirror fallback,
encoder image-token adjustment, overallocated rows, and mismatched inputs.

This does **not** fully verify the original issue. The review host has one AMD
MI355X/gfx950 under ROCm 7.2, not eight NVIDIA B200 GPUs under CUDA, and lacks
the GLM-5.2 744B DSA NVFP4 target and EAGLE draft weights. Therefore the
reported deep-turn cache-hit recovery (40–53% back toward 97%), TTFT recovery,
TP8 behavior, and the separate 256-request CUDA-graph crash remain untested.
The tiny-Llama fixture cannot qualify EAGLE or this model architecture and was
not substituted as proof.

No native files changed. Imports at the candidate resolved to the three source
files under `/job/repo/python/sglang`; no native rebuild was applicable.
Raw issue snapshots, diffs, commands, and logs are retained outside the
revision-switching checkout at `/job/review-evidence-j-d944deffd8db/`.
