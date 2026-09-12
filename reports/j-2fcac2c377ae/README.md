# Correction of PR 2113 after PR 2218 review

Candidate: https://github.com/amdpilot-org/sglang/pull/2113 at `3ac707e08537890da1a90372c58f2718e5b8789a`

Independent review: https://github.com/amdpilot-org/sglang/pull/2218

Upstream issue: https://github.com/sgl-project/sglang/issues/32507

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2253

The candidate's checked-out native source was rebuilt for gfx950 before any correction. It copied N=16 exactly, preserving the valid original fix, but rejected N=513, N=1024, and N=1025 with the new 512-element cap. The production OffloaderV2/DeepEP caller has no corresponding size gate.

The corrected fallback divides an arbitrary positive vector into 512-element by-value kernel arguments. This stays below the kernel-parameter limit, continues to avoid the copy engine, and removes the caller-visible maximum. A separately rebuilt corrected library copied all tested sizes exactly, including the 512/513 boundary, two full chunks at 1024, and a final partial chunk at 1025.

Raw native build and GPU probe output is under `evidence/`. The generated HIP sources and extension caches are retained under `/tmp/amdpilot-repo-j-2fcac2c377ae/`.

The original GLM-5.2 weights, NVIDIA H800s, and two-node TP16/EP16 DeepEP environment were unavailable. This validates the native transfer behavior on one assigned gfx950 GPU; it does not claim the full serving workload, NVIDIA compilation, distributed execution, or model semantics.
