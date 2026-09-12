# Independent review of PR 1015

Upstream issue: https://github.com/sgl-project/sglang/issues/37369

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1050

Candidate: https://github.com/amdpilot-org/sglang/pull/1015 at `72ffcd201344d84479531607c897b31319e90714`

Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

## Recommendation

Accept. The candidate fully resolves the original issue's demonstrated ownership defect. This is a source fix with regression coverage, not merely test hardening.

## Evidence

The prepared checkout was exactly the recorded base. On that base, an independent probe called the actual `Glm5NextForConditionalGeneration.load_weights` implementation with RunAI-marked tensor views backed by a single reused CPU staging buffer. In q-then-kv order the fused value was `[3, 4, 3, 4]`; in kv-then-q order it was `[1, 2, 1, 2]`. Both should have been `[1, 2, 3, 4]`. This reproduces the original corruption mechanism in both arrival orders.

At the exact candidate commit, the same probe returned `[1, 2, 3, 4]` in both orders. The unmarked boundary intentionally continued to alias the reused buffer, showing that the change is scoped to tensors carrying the marker set by `runai_safetensors_weights_iterator`.

The candidate's own GLM-5 regression passed 3 tests. The pre-existing RunAI loader suite passed all 11 tests after the helper and constant were moved, covering the related DeepSeek ownership behavior and marker/loader boundaries. The candidate has no native changes.

Imports were confirmed from `/job/repo/python/sglang/...`; the review did not accidentally test an installed SGLang wheel. Torch resolved from the prepared ROCm environment (`2.11.0+rocm7.2`, HIP `7.2.26015`).

One assigned AMD Instinct MI355X (`gfx950:sramecc+:xnack-`) executed a numerical check using the fused weight produced by the fixed loader. `[[1,2],[3,4]] @ [[2],[-1]]` exactly matched the independent CPU reference `[[0],[2]]`.

## Scope and limitations

No real GLM-5.3-Flash or draft weights, distributed RunAI service, TP=8 topology, Kubernetes deployment, or DFLASH serving setup was available. Consequently this review does not claim full-model generation, semantic accuracy, multi-GPU, or multi-node reproduction. Those limitations do not leave a counterexample to the reported deterministic ownership contract: the marker is applied by the actual RunAI iterator, and every marked GLM-5 shard retained across iterator advancement is cloned by the candidate.

No C++ or other native source changed. The prepared environment declares no native build target, so a native rebuild was not applicable.

Raw outputs, the candidate diff, and captured issue/PR metadata are under `evidence/`.
