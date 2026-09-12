# Independent review of PR 2833 at `bbc1777`

Upstream issue: https://github.com/sgl-project/sglang/issues/37944

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2862

Candidate: https://github.com/amdpilot-org/sglang/pull/2833 at `bbc17774786e3645c45283eb50e960344346200d`

Parent candidate: https://github.com/amdpilot-org/sglang/pull/2706 at `0e4ffed88e34a6c710e94cfb49253c5fbcaa1283`

Parent independent review: https://github.com/amdpilot-org/sglang/pull/2745

## Result

Recommendation: **unverified**. Fully resolves original issue: **false**.

The source-level correction is internally consistent in the cases that can be exercised here. The recorded base reproduces the original static behavior: an 8-token prefill with CP enabled and `cp_size=4` is CP-active. The exact candidate makes the same batch non-CP when `min_tokens=16`.

The candidate also fixes the concrete prior-review counterexample. For DSA interleave with 8 processed tokens and a threshold of 16, generic CP execution, DSA CP execution, and CP-specific DSA metadata sharding all return false. Independent boundary and adversarial cases pass for zigzag and interleave. No additional source-level counterexample was found.

This does not qualify the full original issue. The prepared device is AMD Instinct MI355X on ROCm/HIP 7.2, while this checkout explicitly disables DSA prefill CP on HIP. The GLM-5.2-FP8 weights are unavailable. Therefore no CUDA multi-rank DSA forward, independent numerical comparison, or TTFT benchmark was possible. The candidate should not be represented as fully verified from unit policy tests alone.

No native source changed, so no native rebuild was applicable. Repository imports were confirmed from `/job/repo/python`; Torch was loaded from the pinned prepared environment.

Raw command output is retained under `raw/`. The whole-candidate `git diff --check` reports trailing whitespace only in historical committed raw report artifacts from the parent candidate; the changed runtime source and tests pass a scoped `git diff --check`.
