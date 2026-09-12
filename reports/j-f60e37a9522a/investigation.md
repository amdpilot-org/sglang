# Investigation of EAGLE PD draft graph width mismatch

Upstream issue: https://github.com/sgl-project/sglang/issues/31178

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2291

The prepared base already contains the applicable solution. Upstream PR
https://github.com/sgl-project/sglang/pull/31255, merged as `fc1e3797b77b` on
2026-07-16, added a uniform-width admission invariant to
`EAGLEDraftCudaGraphRunner.can_run_graph`. A runtime batch whose
`spec_info.num_tokens_per_req` differs from the graph's `captured_req_width`
now falls back to eager execution instead of replaying an incompatible graph.

This matches the report's failure mechanism: one request supplied two seed
rows, and replay attempted to copy a `[2, 2048]` source into the graph buffer's
`[1, 2048]` request slice. Temporarily removing only the existing admission
guard makes the added regression fail because the incompatible graph is
admitted. Restoring the guard makes it and the two boundary cases pass.

Raw logs are retained under
`/tmp/amdpilot-repo-j-f60e37a9522a/evidence/` in the prepared runtime. The
source checkout is `/job/repo`; no native library was rebuilt or required.

The assigned AMD Instinct MI355X (`gfx950`) reproduced the exact PyTorch shape
error and validated a matching-shape copy against an independent integer-sum
reference. This is not a full reproduction of GLM-5.2-FP8, TP16, two-node PD
disaggregation, Mooncake transport, or speculative-decoding semantics because
the required weights and topology were unavailable.
