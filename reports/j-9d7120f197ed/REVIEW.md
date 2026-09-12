# Independent review of PR 1110

Candidate: https://github.com/amdpilot-org/sglang/pull/1110 at `090cc57059021edb34acd063d7ba2ab7f3ab8a2d`

Upstream issue: https://github.com/sgl-project/sglang/issues/37852

Mirror issue: https://github.com/amdpilot-org/sglang/issues/3213

## Recommendation

Accept. The candidate fully resolves the original source-level memory-accounting defect in the paths that could be exercised on the assigned gfx950 GPU. It changes the DeepSeek-V4 pools from inherited zero accounting to sums of the physical tensors actually allocated, aggregates those tensors at the top-level pool, and includes the HiSparse C4 `uint64` device-pointer table that the preceding candidate omitted.

This is a functional fix, not merely test hardening. The recorded base reproduced the original zero result with an actual non-unified FP8 allocation: 3,594,944 owned bytes versus 0 reported bytes. At the exact candidate, actual FP8, ROCm FP4-indexer, HiSparse, and unified allocations all had zero difference between independently enumerated owned bytes and `mem_usage`. HiSparse reported 3,594,952 bytes, including its 8-byte one-layer pointer table.

Independent pipeline-partition cases also had zero byte delta for a C4-only HiSparse stage, C128-only stage, SWA-only stage, mixed stage, and unified mixed stage. The enumeration rejected duplicate storage identities so aliased views could not inflate the reference total. The candidate's focused regression passed all six tests and ten subtests, including one- and three-layer HiSparse boundaries.

## Source and environment verification

- Prepared base and failing-before revision: `358c163250ad3b1f62939b01ce1314a0a31a0365`.
- Exact candidate revision: `090cc57059021edb34acd063d7ba2ab7f3ab8a2d`.
- Imports resolved to `/job/repo/python/sglang/__init__.py` and `/job/repo/python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`, so tests used checkout source rather than a wheel copy.
- Interpreter: `/tmp/amdpilot-repo-j-9d7120f197ed/venv/bin/python`; PyTorch `2.11.0+rocm7.2`, HIP `7.2.26015`.
- GPU: AMD Instinct MI355X, `gfx950:sramecc+:xnack-`.
- No native source changed, and `repository-environment.json` identifies no separate native artifact; a native rebuild was therefore not applicable.
- `git diff --check 358c163250ad3b1f62939b01ce1314a0a31a0365..090cc57059021edb34acd063d7ba2ab7f3ab8a2d` passed.

## Limitations

DeepSeek-V4 weights and the reported 8x H100 environment were unavailable. This review therefore does not claim a full HTTP server/metric scrape, CUDA/H100 behavior, multi-rank execution, model semantics, NPU execution, or the CUDA-only online-C128 path. Those are architecture/environment limitations rather than observed counterexamples. The reduced fixture validates real GPU allocations and the exact value consumed by the scheduler metric.

Raw command output is retained in `evidence/`.
