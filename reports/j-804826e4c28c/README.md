# Investigation report: upstream issue 38341

The prepared base already contains the relevant correction. No production
source change is proposed by this job.

## Finding

Upstream PR [#36911](https://github.com/sgl-project/sglang/pull/36911), merged as
`9a05b470fa849b349e384ef3c1381f9a85c6c550` on 2026-09-01, added the DFlash
startup behavior requested by the issue:

- `DFlashWorkerV2.prewarm_sampling()` constructs the largest configured
  non-greedy verify shape after graph capture.
- It first warms process-lifetime buffers outside graph-pool borrowing.
- It then rehearses the verify allocation. If the borrowed graph pool is too
  small, startup catches `torch.OutOfMemoryError`, disables borrowing, and
  repeats the measurement on the normal allocator.
- The measured `sampling_headroom_bytes` is stored on the target model runner.
- `compute_post_capture_kv_resize()` reserves that measured headroom before it
  physically backs the KV pool, so verify workspace and KV are no longer both
  allowed to claim the same bytes.

Upstream PR [#38596](https://github.com/sgl-project/sglang/pull/38596), merged as
`203d7e812c6c9cde8859499daf54686091714638` on 2026-09-10, separately made the
same post-capture budget account for installed KV-canary per-forward workspace,
including the maximum of sequential target and draft runners. The prepared base
contains both merges.

## Regression evidence

The existing regression
`TestGraphPoolBorrow.test_dflash_prewarm_falls_back_when_the_rehearsal_exhausts_the_pool`
asserts the issue-specific transition `False, True, False`: warm without
borrowing, reproduce an OOM while borrowing, disable borrowing, and retry
without borrowing. On the assigned ROCm host, its upstream decorator is
insufficiently specific: `torch.cuda.is_available()` is true under HIP while
`graph_pool_borrow_enabled()` correctly rejects non-CUDA platforms. The
unmodified test therefore fails before entering the intended scenario. Running
the same regression with only the CUDA platform predicate mocked produced:

```text
Graph pool DFLASH sampling rehearsal exhausted graph-pool memory for the
2x4x32 verify probability matrices; disabling borrowing and reserving the
measured headroom in the post-capture KV sizing instead
{'device': 'AMD Instinct MI350X',
 'gcn_arch': 'gfx950:sramecc+:xnack-',
 'calls': [False, True, False],
 'sampling_input_bytes': 1024,
 'sampling_headroom_bytes': 2048}
```

The focused current-source suite passed 26 tests and 27 subtests. Its JUnit
record is in `evidence/focused_passing.xml`. The initial unmodified invocation,
including the ROCm-mis-gated CUDA-only case, is retained in
`evidence/focused_current.xml` (13 passed, one failed for the platform predicate
rather than the feature behavior).

PR #36911's patch is the failing-before/passing-after source history: before
that merge DFlash had no post-capture sampling rehearsal or measured sampling
headroom; the merge added both the implementation and regression. Reverting
that implementation would remove the startup call exercised by the regression,
so the reported mid-traffic allocation would again have no startup rehearsal.

## GPU evidence and boundaries

The assigned device was one AMD Instinct MI350X (`gfx950`, ROCm 7.2). A real-GPU
top-p normalization check used the current DFlash HIP path and compared it with
an independent CPU float64 sort/cumulative-sum/scatter reference:

```text
max_abs_vs_cpu_float64 = 8.738910914352083e-10
max_row_sum_error = 1.1920928955078125e-07
```

For a fixed 64-request, 4096-vocabulary probability tensor, the exact input
storage scaled linearly with draft count: 1, 8, and 16 draft tokens required
1,048,576, 8,388,608, and 16,777,216 bytes respectively. This confirms the
reported workspace scaling boundary, but it is not a reproduction of the
original CUDA/SM80 kernel or distributed model workload.

## Limitations

- The reported 2x8 A800, TP8, mooncake-RDMA PD deployment and its target/draft
  weights were unavailable.
- SGLang explicitly reports the CUDA DFlash sampling-verify kernel unavailable
  on this ROCm/gfx950 host. Therefore the original CUDA kernel OOM and HTTP 500
  traffic failure were not reproduced here.
- A tiny Llama serving fixture would validate only transport and generic engine
  execution; it cannot represent the reported Qwen4 GDN/MoE architecture,
  DFlash draft model, TP8, or PD memory pressure, so it was not used as a claim
  about this issue.
- No native library was changed or rebuilt.

