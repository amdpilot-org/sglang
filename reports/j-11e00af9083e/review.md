# Independent review of PR 1065

Candidate: https://github.com/amdpilot-org/sglang/pull/1065 at `342a1a9091a7b7d9ec220c4d6a877280b8543280`

Upstream issue: https://github.com/sgl-project/sglang/issues/39173

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1099

Recommendation: **request changes**. The candidate is a useful partial fix, but it does not fully resolve the original Engram compact-ragged contract.

At the exact recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`, the actual layout builder reproduced the original 42-token/eight-slot geometry: `[6,6,5,5,5,5,5,5]`. On the assigned MI350X, both DSV4 raw verify and decode metadata copies also reproduced the reported overlapping-storage exception.

At the exact candidate, the focused regression suite passed (14 tests and 8 subtests). The 42-token tier now correctly uses seven width-six rows, and both overlapping metadata-copy paths pass on the real GPU. These are valid source fixes, not merely test hardening.

The candidate nevertheless leaves compact-ragged tiers that an equal-block-only Engram consumer cannot process: 41 tokens produce `[6,6,6,6,6,6,5]`, 43 produce `[6,6,6,5,5,5,5,5]`, and 47 produce `[6,6,6,6,6,6,6,5]`. The candidate test comments acknowledge that Engram must use per-request layout, but neither the prepared source nor the candidate contains `EngramHasher` or `engram.py`. There is therefore no executable Engram path or independent numerical reference with which to prove correct per-request hash context and output.

The original four-node NVIDIA GB10 TP4/EP4 DeepSeek-V4.1-Flash setup and weights are unavailable. The later first-forward fix was verified only at its isolated overlapping-tensor mechanism on one AMD Instinct MI350X under ROCm 7.2, not through full model capture or serving. No native source changed, so a native rebuild was not applicable. The tiny Llama fixture was intentionally not used as proof for this different architecture and contract.

Raw failing-before and passing-after evidence is retained in `reports/j-11e00af9083e/raw/`.
