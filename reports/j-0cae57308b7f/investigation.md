# Independent review of PR 2138

Upstream issue: https://github.com/sgl-project/sglang/issues/33360

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2174

Candidate: https://github.com/amdpilot-org/sglang/pull/2138 at
`29db02e58ef3d9b268bf70dbb5903a247b76f7be`.

Parent candidate: https://github.com/amdpilot-org/sglang/pull/1958 at
`152efc1aaa4da7cf62be370847166446667d21e2`.

Parent independent review: https://github.com/amdpilot-org/sglang/pull/2045.

## Recommendation

Accept PR 2138 as **test-only hardening**. It makes no production or native
change, so it is not itself a full fix for the original serving issue. The
recorded base already contains the production gather correction. The candidate
correctly closes the two concrete oracle gaps reported against PR 1958:

- it requires the complete local token operand
  `input_ids[:, None].clone()` in both the main and NextN forwards; and
- it rejects any `dp_gather_partial` call in `_run_moe_ffn_dp_sync` while
  preserving the independently partial TBO `op_gather_a` boundary.

## Independent reproduction

At exact parent commit `152efc1`, its regression passed a fixture containing
all three review corruptions: both token gathers used
`input_ids_global.clone()`, and `_run_moe_ffn_dp_sync` contained an additional
`dp_gather_partial`. Result: `3 passed`, exit 0. This independently reproduces
the review of PR 1958.

At exact candidate commit `29db02e`, the clean candidate test passed (`3
passed`). The combined corrupted fixture failed (`2 failed, 1 passed`). Four
isolated fixtures were then tested to avoid short-circuiting in the two-file
token loop:

- wrong main-model token operand: `1 failed, 2 passed`;
- wrong NextN token operand: `1 failed, 2 passed`;
- extra synchronous partial gather: `1 failed, 2 passed`;
- reconstructed historical partial gather plus uncloned token views:
  `2 failed, 1 passed`.

Thus the candidate rejects each concrete counterexample independently. The
recorded base itself has the corrected production source, so the historical
failure was reproduced with a source fixture restoring those exact bad
expressions rather than falsely claiming that the prepared base still fails.

## Paths, GPU, and native code

The test interpreter imported `sglang` from
`/job/repo/python/sglang/__init__.py`. Torch was `2.11.0+rocm7.2` with HIP
`7.2.26015`. The assigned GPU identified as AMD Instinct MI350X, capability
reported by Torch as `(9, 5)` (gfx950 family).

A real GPU arithmetic/aliasing reference showed that summing identical
replicas scales `[1.25, -2.0, 3.5]` by widths 2, 4, and 8, selecting a replica
preserves it, zeroing `input_ids[:, None]` mutates the source IDs, and zeroing
its clone preserves `[7, 11, 13]`. This validates only the numerical mechanism.

The candidate changes only Python tests and reports. There is no native source
change and no native rebuild was required. The environment metadata reports no
prepared native artifact (`native: null`, `wheel: null`).

## Remaining limitation

The original eight-NVIDIA-H800, TP8/DP4, DeepSeek-V4-Flash-0731 Marlin serving
case could not run on one AMD MI350X without the model weights, CUDA, H800s, or
Marlin. Generated-text semantic accuracy therefore remains unverified. This is
the sole remaining original-contract counterexample and is not converted into
a speculative source change.

Raw command output and revision-independent fixtures were preserved during
checkout switching under `/job/review-evidence-j-0cae57308b7f/`.
