# Independent review of PR 3003

Upstream issue: https://github.com/sgl-project/sglang/issues/7892

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/2964

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/3037

Candidate: https://github.com/amdpilot-org/sglang/pull/3003 at `ec7e5ca1467413eb2487812d44f1d10c96cb42f3`

## Recommendation

`unverified`; `fully_resolves_original=false`.

The patch is a credible partial fix for a deliberately narrow configuration: eager, fixed-width DeepSeek NextN draft-extend using segmented A2A, without context parallelism. Its focused tests pass, and an independent one-GPU numerical check confirms that its tensor splitting, index rebasing, and output-buffer views are exact.

That evidence does not establish the original issue's end-to-end contract. The candidate's own meaningful integration test requires four GPUs and DeepSeek target/NextN model weights. It was only collected here. Consequently, this review did not observe a real draft-extend batch entering the two interleaved pipelines, distributed ranks completing matching collectives, semantic/acceptance equivalence, or the claimed increase in running batch size.

The change also leaves several draft-model configurations outside the feature: CUDA-graph draft-extend, widened/multi-layer inputs, non-segmented dispatch, context parallelism, and architectures other than `DeepseekV3ForCausalLMNextN`. Those are explicit gates/assertions rather than merely missing tests, so the candidate cannot be called a full resolution of the broadly stated open issue.

## Evidence

- Recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`: the extracted candidate regression failed during collection because `_compute_moe_deepseek_draft_extend` does not exist.
- Exact candidate commit: 51 related unit tests and 7 subtests passed.
- Independent GPU adversarial check: 35 batch-size/request-width combinations passed exactly on one AMD Instinct MI350X (`gfx950`), including odd batches, width one, flat-index rebasing, and capture-buffer view writes.
- The three candidate end-to-end cases collected successfully but were not run.
- Imports resolved to `/job/repo/python/sglang/...` at the candidate commit.
- No C++/FlyDSL/native sources changed; the prepared environment reports `native=null`, so no native rebuild was applicable.

Raw logs and the preserved candidate diff are under `/job/review-evidence-j-418b6b8f7292`, outside the checkout used for revision switching.

## Environment boundary

Only one GPU was assigned, and no qualifying DeepSeek-V3/NextN weights were available. The deterministic tiny Llama fixture is not a substitute because it cannot exercise the candidate's model-class gate, DeepSeek NextN layer, segmented A2A, or four-rank collective schedule.
