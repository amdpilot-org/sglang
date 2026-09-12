# Independent review of candidate PR 2780

Reviewed exact candidate commit `333b37951dfca5d98dffb77b25ee5a8e0801100b` against recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365` and the complete Level 1 contract in the upstream issue.

Recommendation: **request changes**. The implementation is substantial, but it does not yet qualify as a full original-issue fix.

## Findings

1. Request admission does not enforce explicit `temperature=0`. `SamplingParams(temperature=0.7, top_k=1)` passes because the guard checks only normalized `top_k`. This is a direct counterexample to the Level 1 request contract.
2. Beam search is not rejected. `SamplingParams(temperature=0, beam_width=2)` passes admission, although the supported shape is greedy, one proposal, top-k 1, and one speculative step.
3. The exact checkpoint contract is not enforced. Shape-compatible arbitrary target and assistant identifiers and mutable revisions pass server validation. The Level 1 supported shape names exact repositories and immutable target/assistant revisions.
4. Runtime correctness remains unverified. The prepared host is Linux x86_64 with AMD ROCm and cannot import `mlx`. Eleven MLX-dependent Stage A cases and the complete pinned Stage B server test skipped. Consequently there is no real evidence here for Metal execution, exact-token parity, rotating MLX cache behavior, proposal/verification counters, cleanup, flush, or post-flush serving.

## What did pass

The architecture-independent candidate suite passed with `38 passed, 11 skipped, 26 subtests passed`. The pinned Stage B test collected correctly and skipped. Changed MLX Python sources compiled, and `git diff --check` passed. No native C++ or FlyDSL source changed, so native rebuilding was not applicable.

Raw command output is retained in `reports/j-0044850f94ad/raw/`.
