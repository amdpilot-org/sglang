# Independent review of singleton grammar token synchronization

Upstream issue: https://github.com/sgl-project/sglang/issues/35826

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1368

Candidate PR: https://github.com/amdpilot-org/sglang/pull/1338

Candidate commit: `cf963c0763d40b42918accb80f2c230984f52f2b`

Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

Recommendation: **accept**. The candidate fully resolves the source-level contract in the original issue.

On the recorded base, both grammar-triggered and environment-forced token synchronization submit `torch.distributed.all_reduce(..., ReduceOp.MIN, group=tp_sync_group)` even when the mocked selected group has world size one. The inactive path does not submit a collective.

At the exact candidate commit, the candidate regression passes. Independent checks show that sizes zero and one skip the collective, size two preserves the exact selected group and `ReduceOp.MIN`, and an inactive/empty-grammar request does not query group size. A real world-size-one NCCL-backend process group on the assigned AMD Instinct MI350X (gfx950, ROCm 7.2) also skips the trapped collective and leaves the GPU token tensor unchanged.

The imported sampler source was `/job/repo/python/sglang/srt/layers/sampler.py`, confirming that tests exercised the checked-out source. The change is Python-only; no native source or generated native artifact changed, so no native rebuild was applicable.

The original 8x NVIDIA B300, CUDA 13.0, NCCL 2.28.9 topology and its reported 512 MiB allocation could not be reproduced on the assigned single AMD GPU. The review therefore verifies the defective call-site behavior and its correction, not the NVIDIA allocator trace, a full serving configuration, or a multi-node workload.

The candidate's bundled report names a different mirror issue (`/issues/1291`); that documentation error does not affect the source fix reviewed here.
