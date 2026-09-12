# Independent review of PR 2367

Upstream issue: https://github.com/sgl-project/sglang/issues/31133

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/2292

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2406

Candidate: https://github.com/amdpilot-org/sglang/pull/2367 at `5c8df7c6487d30fc14449489e7c3e83972778e82`

Recommendation: **accept**. The candidate fully resolves the original allocation-safety bug for the exercised kernel contract.

The recorded base reproduced the defect on the assigned MI350X/gfx950 GPU: the stale-width call produced 163 mismatched top-k elements out of 1040 versus the correctly sized call. On the exact candidate, all three submitted regression cases passed. Six independent cases also passed, including block-boundary underestimates, a zero-width caller, same-rounded/oversized controls, and exact comparison of both attention output and top-k indices.

The loaded implementation was confirmed at `/job/repo/python/sglang/kernels/ops/attention/minimax_sparse/prefill/flash_with_topk_idx.py`. No native source changed, so no native rebuild was applicable. Raw logs and the independent harness were preserved during revision switching under `/job/review-evidence-j-e76f2cab2baf/`.

The backend's stale host metadata source remains, but it is no longer trusted for allocation safety: the candidate obtains the required block width from the live `seq_lens` tensor used by the kernels and emits a warning when it enlarges the allocation. No remaining counterexample was found.

Limitations: this was single-GPU AMD gfx950 validation, not GB200/SM100 TP=4 MiniMax-M3 serving. CUDA compute-sanitizer, Xid/NCCL failure behavior, production model semantics, and distributed execution remain unverified.
