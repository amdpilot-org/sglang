# Independent review of diffusion startup profiling

Candidate reviewed: https://github.com/amdpilot-org/sglang/pull/3188 at exact commit `ef9d09bb5480f35772748d2c802def6b62929cd0`.

Upstream issue: https://github.com/sgl-project/sglang/issues/19087

Mirror issue: https://github.com/amdpilot-org/sglang/issues/3190

## Recommendation

Accept. The exact candidate fully addresses the original request and the four concrete counterexamples from the previous independent review.

On the recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`, the candidate regression could not collect because `startup_profiler` did not exist. At the exact candidate, all 11 candidate tests passed. Three separately written adversarial tests also passed and exercised:

- the real worker entry with profiling disabled, proving the former unconditional undefined-name failure is gone;
- the console clock being established before the lazy `sglang.cli.generate` import;
- task pipes, result pipes, scheduler-pipe/process construction, worker start, parent cleanup, ready wait, and worker IPC snapshot appearing in one parent-owned tree.

The existing launch shutdown suite also remained green (3 tests). Imports resolved from `/job/repo/python`, not an installed SGLang wheel. The patch changes Python only, so no native rebuild is applicable.

The candidate history retains a successful Qwen-Image launch on an AMD Instinct MI355X at exact model revision `75e0b4be04f60ec59a75f475837eced720f823b6`, including component timings and a generated image checksum. This review did not repeat that model launch: the current assigned GPU is an AMD Instinct MI350X (`gfx950`) and the Qwen-Image weights are absent from this job's private runtime. The deterministic correction paths execute before model inference and were independently exercised here. Multi-GPU, multi-node, disaggregated, CUDA, Intel, Apple, and Ascend paths remain unverified rather than being treated as failures.

Raw review evidence was preserved outside revision switching under `/job/review-evidence/j-f8e03fe4ed2d/`.
