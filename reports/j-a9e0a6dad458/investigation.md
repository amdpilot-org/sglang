# Consolidated correction for DSA kpool logits chunking

Upstream issue: https://github.com/sgl-project/sglang/issues/37712

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1119

Candidate PR: https://github.com/amdpilot-org/sglang/pull/985 at
`a4c8fbf3a96e6e935572f45d581cb8554514c09a`

Independent review PR: https://github.com/amdpilot-org/sglang/pull/1084

## Finding

The candidate correctly activated the previously unused logits-memory budget and
preserved the fitting fast path, but both reviewed counterexamples reproduce.
Its query-only loop always retains the entire concatenated K axis, so the
smallest fp32 output is one `total_k_rows` row and may exceed the budget. In
fused PAGED mode it also slices the global request page table by query-row
offsets while retaining global request IDs.

The correction records each request's Q/K slice while the host plan geometry is
available. Only the chunked path changes: it computes each request against its
own K window, rebases K-relative indices, retains the full global PAGED table,
and slices only row-indexed metadata. This preserves the candidate's budget
selection and dense fast path while making a one-row chunk proportional to one
request rather than the entire batch.

## Evidence and limitations

`candidate-counterexamples-before.log` is an exact-candidate GPU orchestration
run: both cases fail. The RAGGED case creates a 48-byte logits row against a
24-byte budget. The PAGED case reaches a later chunk whose table view no longer
has the original global base while its row indices remain global.
`counterexamples-after.log` records 2/2 passing, and `regression-after.log`
records 11/11 passing across the candidate, counterexample, and existing budget
suites.

The assigned device was one AMD Instinct MI350X (`gfx950`) with ROCm 7.2.
Real fp8/device tensors and production Python orchestration were exercised;
DeepGEMM and top-k were mocked because NVIDIA DeepGEMM is unavailable here.
No GLM-5.3-Flash/RadixArk weights or four-B300 TP/EP environment were available,
so the original serving workload and semantic model output remain unverified.
