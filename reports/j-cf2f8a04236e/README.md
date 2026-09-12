# Independent review of PR 2642

Upstream issue: https://github.com/sgl-project/sglang/issues/3050

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2645

Candidate: https://github.com/amdpilot-org/sglang/pull/2642 at exact commit `2a7700a661f85826b911af4cd096ac79ef5185b9`.

## Recommendation

Accept. The candidate fully resolves the original issue as a metric-scope clarification. The reported values are different estimators, rather than evidence of an erroneous throughput formula. The candidate does not change metric calculations; it corrects the benchmark's user-facing explanation so it accurately states all three relevant boundaries:

- TPOT excludes TTFT and covers the post-first-token decode phase.
- ITL samples are intervals between nonempty stream events, need not be one per output token, and can omit a terminal response tail.
- Client output throughput is a whole-run aggregate, while engine generation throughput is a recent scheduler-window aggregate; exact agreement is not expected.

On the recorded base, an independent synthetic request reproduced the confusing state: TPOT was 100 ms while whole-lifetime time per output token was 490 ms, eight ITL samples represented nine post-first output tokens and left a 0.1 s terminal tail, and whole-run throughput was 2.0408 tok/s while an independently selected recent window was 20 tok/s. The base had no scope explanation. At the exact candidate, its two regression tests passed and the same independent adversarial case confirmed that the new note describes the implementation correctly.

This is a source/documentation correction, not a native change. The imported benchmark module resolved to `/job/repo/python/sglang/benchmark/serving.py`; no C/C++/CUDA/HIP files changed, so no native rebuild was applicable.

## Limitations

The review host has one AMD Instinct MI350X (`gfx950`) with PyTorch 2.11.0+rocm7.2 and ROCm 7.2.26015, not the issue's NVIDIA H800/CUDA stack. Qwen2.5-0.5B weights and the original 2048-input/256-output, concurrency-16 serving workload were not run. These limitations prevent performance-number equivalence claims, but do not block verification of the estimator definitions or the candidate's explanatory change. The candidate's retained tiny-Llama serving evidence is transport/engine evidence only and was not treated as proof for Qwen or H800 performance.

`git diff --check` reports trailing whitespace in committed review log artifacts under `reports/j-dc5d36276828/`; this is nonfunctional evidence-file hygiene and does not affect the recommendation.

Raw review outputs are retained under `evidence/`.
