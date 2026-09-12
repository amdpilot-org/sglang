# DSA top-k v2 tie-overflow correction generation 1

Upstream issue: https://github.com/sgl-project/sglang/issues/35257

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1599

Candidate PR: https://github.com/amdpilot-org/sglang/pull/1483

Independent review PR: https://github.com/amdpilot-org/sglang/pull/1561

The prepared base and exact candidate commit
`d501752c27de267fbc85dce32832d2a59cdba8bd` were checked independently. On
the base, deterministic rows with more than 2048 fp32-distinct values in one
coarse threshold bin returned 2048 valid unique indices but the wrong selected
value multiset. The first overflowing boundary (2049 candidates) also failed.
At the exact candidate, all register and streaming cases matched `torch.topk`.

Source inspection confirmed the review's remaining counterexample: CUDA routes
the original batch-1, N=262144 shape to `TopKCluster`, where each rank truncates
its local arrival-order candidate set and the primary rank ultimately calls
`handle_tie` with `min(equal_count, kMaxNumTie)`. The assigned gfx950 cannot
execute CUDA thread-block clusters, so no unvalidated cluster-wide algorithm was
introduced. Instead, this correction disables CUDA cluster dispatch and sends
those rows through the candidate's exact streaming refinement. This trades
long-row CUDA performance for correctness until a cluster-wide refinement can
be implemented and tested on supported hardware.

A dedicated batch-1, N=262144 regression uses 50,000 fp32-distinct values in
one coarse bin. The same mechanism failed before (the independent base harness
reported 2048/2048 value mismatches at N=262144) and passes after on gfx950.
The complete top-k v2 test file passes 289 tests against a freshly built private
JIT library.

CUDA/B200 execution, GLM-5.2 weights, the reported score dumps, TP8, and a
multi-node workload were unavailable. Consequently the CUDA dispatch change is
source-verified but not performance- or runtime-validated on CUDA hardware.
