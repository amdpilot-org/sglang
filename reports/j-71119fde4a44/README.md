# Independent review of PR 1093

Candidate: `a6e24ec6b9964325bc475561c7040235a4cd8f2f`

Upstream issue: https://github.com/sgl-project/sglang/issues/36886

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1124

## Finding

The candidate is a partial, component-level fix, but it is not sufficient to
claim that the full original serving issue is resolved. On the recorded base,
the no-rope MLA DCP writer reproducibly writes raw virtual locations; on the
candidate, its new owner/localization regression passes on the assigned AMD
gfx950 GPU. The candidate's index-K virtual-capacity and DCP assertion unit
tests also pass.

An independent adversarial GPU case found a remaining defect in the changed
no-rope path. `set_mla_kv_buffer_dcp_sharded_triton` defaults to
`reserved_skip_index=0`, and the common dispatcher documents that reserved
writes are skipped. The rope kernel includes that condition, but the changed
no-rope kernel does not. On DCP rank 0, a padding write at virtual location 0
therefore overwrites physical slot 0; a NaN padding row was observed to make
that slot NaN. Rank 1 correctly filters the same location only incidentally by
the owner rule. The candidate regression disables skipping with
`reserved_skip_index=-1`, so it cannot detect this counterexample.

The prepared base has newer unified-pool and `KVIndexTranslator` machinery
than the historical code quoted by the issue. This review therefore evaluated
the actual prepared implementation and candidate. No model weights and only
one GPU were available. The eight-rank H100 GLM-5.3-Flash watermark crossing,
sparse decode/verify LSE merge, sparse extend gather, semantic quality, and
long-running allocator behavior remain unverified.

## Evidence

Raw logs are retained outside the checkout at
`/job/review-evidence-j-71119fde4a44/` so they survived revision switching.
The base failure uses the candidate's test unchanged and only in-bounds
locations, avoiding undefined OOB behavior. The independent adversarial test
also checks localization at the physical-capacity boundary and the reserved
slot on both simulated DCP ranks.

No native C++ or FlyDSL source changed. The imported `sglang` source was
`/job/repo/python/sglang`, and the tests compiled/executed the changed Triton
kernel from that checkout. A native rebuild was therefore not applicable.
