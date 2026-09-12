# Investigation report

Upstream issue: https://github.com/sgl-project/sglang/issues/39072

Mirror issue: https://github.com/amdpilot-org/sglang/issues/735

The reported exception is a direct shape invariant failure: FlashMLA requires
`num_splits.shape == (query_batch + 1,)`. In the eager Eagle path,
`draft_attn_backend.init_forward_metadata(forward_batch)` runs before
`ModelRunner._prepare_eager_forward_batch()` calls `prepare_mlp_sync_batch()`.
The former invokes `pad_dsa_cache_seqlens`; the latter aligns the query batch to
`attn_tp_size`. Consequently, raw per-DP counts such as 9 can pre-plan 9 DSA
rows, then become 12 query rows for attention TP 4.

The correction applies the same attention-TP alignment while calculating the
DSA metadata size. It is deliberately idempotent when preparation has already
updated `global_num_tokens_cpu`.

Related change review found upstream PR
https://github.com/sgl-project/sglang/pull/30642, which proposes the same
alignment for an earlier version of this helper. Current `main` had since been
refactored to reuse `dp_padding_mode`, but no longer performed the alignment in
the pre-plan case. The issue-specific regression fails on the recorded base
with 9 rather than 12 rows and passes after this patch. It also covers the
other DP rank, MAX_LEN, and the no-padding boundary.

The full report configuration was not reproduced: the job has one AMD gfx950,
not eight NVIDIA H20s, and does not have the GLM-5.3 weights, CUDA FlashMLA,
Mooncake/IB, or a distributed allocation. The GPU probe only validates that the
production padding function performs the expected shape transformation on the
assigned device; it does not qualify model accuracy or the serving topology.

Raw issue, related-PR, test, and GPU outputs are retained under `raw/`.
