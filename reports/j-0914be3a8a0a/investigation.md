# Candidate correction investigation

Upstream issue: https://github.com/sgl-project/sglang/issues/30609

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2536

Candidate PR: https://github.com/amdpilot-org/sglang/pull/2426

Independent review PR: https://github.com/amdpilot-org/sglang/pull/2508

Exact candidate commit: `b8902dbde7a9d707fc5729883551ed082c110baa`

Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

## Result

The candidate's Mooncake batching mechanics are valid, but its change from an
opt-in limit of `0` to a global default of `1024` is not justified by the
available evidence. This correction therefore preserves the ordered batching
implementation already present on the base and adds a regression that keeps
the model-dependent limit opt-in.

The exact candidate passed its batching suite (6 tests and 3 subtests), so its
index slicing, ordering, explicit-zero behavior, failure propagation, device
indices, and custom-memory-pool boundary remain credible. However, the original
GLM-5.2-FP8 deployment was not reproduced by the candidate or here. Upstream PR
32758 explicitly reports that FP8 did not reproduce at concurrency 8; its
successful 1024-index validation used `nvidia/GLM-5.2-NVFP4`, separate 4x L20D
prefill/decode nodes, and a different workload. That PR also states that other
models can have different bytes per KV index.

The original issue's H100 topology, two 2-node prefill instances, 4-node/32-rank
decode instance, Mooncake/RDMA, DSA, HiCache, HiSparse, and DeepEP combination
was unavailable. A follow-up in the source issue reports that adding
`--moe-a2a-backend deepep` causes a freeze while removing it does not. A
Mooncake batch-size default cannot independently establish a correction for
that path.

## Evidence

Against the exact candidate, the new opt-in invariant fails because the
configured default is `1024`. Against this correction it passes with default
`0`, together with all existing batching tests. This is the relevant
failing-before/passing-after regression; it does not claim an end-to-end
reproduction of the original distributed failure.

Raw API snapshots and logs are retained in
`/tmp/amdpilot-repo-j-0914be3a8a0a/evidence/`. The prepared interpreter was
`/tmp/amdpilot-repo-j-0914be3a8a0a/venv/bin/python`, importing source from
`/job/repo/python`. No native code changed and no native artifact was provided.

The available accelerator is one AMD Instinct MI355X (`gfx950`) under ROCm
7.2, not an NVIDIA H100 cluster. It was inventoried but not used for an
unrelated smoke test, because that would not exercise Mooncake/RDMA, DeepEP, or
the reported model and topology.
