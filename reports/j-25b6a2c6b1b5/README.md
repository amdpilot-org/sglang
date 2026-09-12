# Review of PR 1483 at d501752c27de267fbc85dce32832d2a59cdba8bd

Upstream issue: https://github.com/sgl-project/sglang/issues/35257

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1518

Candidate: https://github.com/amdpilot-org/sglang/pull/1483

Recommendation: **request changes**. The candidate is a verified partial fix,
not a full resolution of the original issue.

On recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`, an independent deterministic
harness reproduced silent value loss while the kernel still returned 2048 valid,
unique indices. The first overflowing boundary (2049 candidates) failed, while
2048 candidates and a 5000-way exact tie passed. Register and streaming cases
with distinct fp32 values lost the true top-k value multiset.

At exact candidate commit `d501752c27de267fbc85dce32832d2a59cdba8bd`, a
fresh private JIT build passed all independent cases and the candidate's 10
narrow-bin regression cases on the assigned MI350X/gfx950. The imports resolved
to the prepared checkout, and the rebuilt shared library was created under the
private runtime cache; paths and timestamps are retained in `evidence/`.

The blocking gap is visible in the implementation itself: refinement is called
only from `TopKRegister` and `TopKStreaming`. `TopKCluster`, guarded by
`#ifndef USE_ROCM`, retains the old truncation and even has a new comment saying
so. CUDA dispatch sends the issue's batch-1, 262144-element B200 shape to this
cluster path. Thus the candidate does not fix the reported architecture/path.

This host has one AMD Instinct MI350X (gfx950), ROCm/HIP 7.2, and no CUDA/B200.
It also lacks GLM-5.2 weights, real dumped score rows, and TP8/multi-node
resources. Those limitations prevent execution of the remaining CUDA
counterexample and any full-model validation; they do not erase the source-level
counterexample.

Raw outputs and the standalone review harness are under `evidence/`.
