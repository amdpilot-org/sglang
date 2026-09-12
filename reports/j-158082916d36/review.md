# Independent review of PR 2108

Upstream issue: https://github.com/sgl-project/sglang/issues/32527

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/2049

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2145

Candidate: https://github.com/amdpilot-org/sglang/pull/2108 at `907c7dc1e99455a48716e31730101aad1e2e08e7`

## Recommendation

Accept. The candidate fully resolves the original source-level deadlock mechanism. The recorded base chooses CUDA graph replay for a rank with a DSA seed and eager execution for a rank without one. The candidate makes that choice rank-independent in the affected PD-decode/attention-DP/DSA-sharing configuration by putting every rank on eager execution.

This is a correctness fix, not merely test hardening. The eager path calls `prepare_mlp_sync_batch`, which derives padding and collective buffer geometry from the globally shared token-count vector. For the issue's one-active/seven-idle pattern, the independent check selected `SUM_LEN` and a shared buffer length of four on every rank, avoiding the graph/eager mismatch.

## Evidence

- The base regression passed three subtests demonstrating the original seed-present graph versus seed-absent eager divergence.
- At the exact candidate, the EAGLE unit file passed 9 tests and 10 subtests.
- At the exact candidate, the disaggregation wire suite passed 37 tests and 12 subtests.
- An independent external harness passed 8 adversarial cases. Unlike the candidate regression, it explicitly exercised `ForwardMode.IDLE` as well as non-idle decode ranks.
- Imports resolved to `/job/repo/python/sglang/srt/speculative/eagle_worker_v2.py`; Torch was `2.11.0+rocm7.2` from the prepared environment.
- A real AMD Instinct MI355X (`gfx950`) executed the numerical DP-count check and matched the CPU reference.
- No native source changed and the prepared environment records no native component, so no native rebuild was applicable.

Raw logs and JUnit files were preserved in `/job/review-evidence-j-158082916d36` while revisions were switched.

## Limitations

Only one gfx950 GPU was assigned. The reported deployment used eight B30Z Blackwell GPUs, and GLM-5.2-FP8 weights plus a full EAGLE PD prefill/decode deployment were unavailable. Therefore the exact multi-rank NCCL hang, model semantics, and the performance cost of disabling draft graphs in this narrow configuration were not measured end-to-end. These are validation limitations rather than observed counterexamples; no remaining source-level counterexample was found.
