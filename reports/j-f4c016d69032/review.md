# Independent review of PR 3210

Candidate: https://github.com/amdpilot-org/sglang/pull/3210 at `c1723a1a501e1241bfaceaac0480c6ae86d2fdff`

Upstream issue: https://github.com/sgl-project/sglang/issues/1763

Mirror issue: https://github.com/amdpilot-org/sglang/issues/3225

Recommendation: **request changes**. The candidate is a partial implementation, not a full resolution of the original request.

## Findings

1. The SM100 correction is valid. The pinned dependency's public `sageattn` dispatcher has no `sm100` branch and raises `ValueError`. The candidate rejects SM100 both during argument resolution and defensively during backend construction.
2. The exact documented dependency has another architecture counterexample. In SageAttention commit `d9704247a5139ab4c03bf7fc6b35cc0e2cbb5ea4`, `SUPPORTED_ARCHS = {"8.0", "8.6", "8.9", "9.0", "10.0" "12.0"}` evaluates to a set containing `"10.012.0"`, not `"10.0"` or `"12.0"`. Consequently its source-build target selection rejects SM120, although the candidate admits SM120 and its documentation identifies only SM100 as excluded. The candidate report's statement that the build script recognizes SM100 is also inaccurate for this exact revision.
3. The candidate's GPU regression uses PyTorch SDPA as a replacement callable. It checks SGLang tensor layout, scale, causality, and GQA expansion, but it neither imports SageAttention nor executes quantized INT8 Q/K attention. This is useful adapter regression coverage, not numerical evidence for the original accuracy claim.
4. The assigned MI350X/ROCm node cannot build or execute the CUDA-only dependency. No supported NVIDIA build/import path, ISA, or runtime kernel was verified.
5. There is no end-to-end serving, quality, long-context, throughput, latency, or memory result. The original approximately 2x motivation remains unverified, and the dense paged-KV gather plus materialized GQA/MQA expansion remains an unmeasured cost.

## Classification

- SM100 handling: fixed and regression-tested through platform overrides; source evidence independently confirmed.
- Experimental adapter wiring: present; Python paths and CLI choice confirmed.
- Architecture compatibility: partial, with the remaining SM120 source-install counterexample above.
- Quantized accuracy and speed contract: unverified.
- Candidate tests: pass, but the numerical portion is test-only hardening around a substituted full-precision callable.

Raw command output was retained outside the temporary candidate checkout during revision switching. The concise evidence used for the committed result is in `evidence.txt`.
