# Investigation: DSA kpool ragged MQA-logits OOM

Upstream issue: https://github.com/sgl-project/sglang/issues/37712

Mirror issue: https://github.com/amdpilot-org/sglang/issues/918

## Finding

The defect is present at base `358c163250ad3b1f62939b01ce1314a0a31a0365`.
`IndexerKPool._get_topk_ragged_kpool_plan` invokes
`deep_gemm.fp8_mqa_logits` once for a dense fp32
`[sum(query_rows), sum(pooled_kv_rows)]` result. The adjacent
`_should_chunk_mqa_logits` helper was never called. For the independently
reported 8192 x 348032 shape this is a 10.62 GiB output allocation; larger
multi-request batches can produce the reported tens-of-GiB request even when
the KV pool itself is not full.

Related upstream work was inspected before implementation:

- https://github.com/sgl-project/sglang/pull/38469 is an open, broader candidate
  which groups by request and then chunks query rows.
- https://github.com/sgl-project/sglang/pull/38507 independently validated the
  narrower query-row chunking approach on 8x H20 and was closed in favor of
  #38469.
- https://github.com/sgl-project/sglang/pull/36960 is already merged but only
  caps the non-kpool ROCm/AITER path; it does not repair this plan call site.

This change wires the existing kpool budget into the affected plan path,
chunks independent query rows, slices all row-indexed metadata, releases each
logits chunk before the next allocation, and preserves the original one-call
path when the matrix fits.

## Evidence and limitations

`regression_before.log` records the new regression against an archive of the
base commit: 5 failed, 1 passed. In particular, forcing a two-row budget still
made one DeepGEMM call instead of three. `regression_after.log` records 6/6
passing on the prepared checkout.

The assigned GPU was an AMD Instinct MI355X (gfx950), ROCm 7.2, exposed as one
device. The GPU regression uses real fp8/device tensors through the actual
kpool plan implementation, but mocks DeepGEMM and top-k to isolate and measure
chunk dispatch and metadata slicing. No GLM-5.3-Flash weights were available.
The reported 4x B300 CUDA/TP serving workload, prefix-cache traffic, semantic
model accuracy, and NVIDIA DeepGEMM allocation were therefore not reproduced
locally. The result is a candidate fix supported by deterministic regression
evidence and the independent H20 validation above, not a claim of full model
reproduction on this AMD fixture.

