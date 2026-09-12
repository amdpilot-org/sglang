# Investigation evidence

Upstream issue: https://github.com/sgl-project/sglang/issues/30609

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2372

The recorded base already includes commit `a6b542813ff46de8ba3856e0236f1af35f4fca54`
(`fix(glm-5.2-nvfp4): bound Mooncake synchronous transfer batches`, upstream PR
https://github.com/sgl-project/sglang/pull/32758). That change added ordered index
batching and tests after reproducing Mooncake session failure followed by decode-side
`KVTransferError` on a long-context GLM-5.2-NVFP4 PD workload.

However, the prepared implementation configured
`SGLANG_MOONCAKE_MAX_TRANSFER_BATCH_INDICES` as `0`, disabling batching unless the
operator supplied an environment override. The original issue's commands contain no
such override. The related PR's measured 1024-index batches reduced a roughly 30.30 GB
main-KV synchronous call into roughly 2.49 GB ordered calls and completed 16/16 requests.
This patch makes that already-tested bound the default; explicit `0` still restores the
legacy behavior.

Raw command output is retained outside the worktree under
`/tmp/amdpilot-repo-j-d116ec16d7c3/evidence/`. The tested source is `/job/repo`; the
prepared interpreter is `/tmp/amdpilot-repo-j-d116ec16d7c3/venv/bin/python`. There is no
native library for this checkout and no native code was changed.

The available device is one AMD Instinct MI355X (gfx950), not the reported H100 fleet.
The GLM-5.2 weights, Mooncake/RDMA peers, and multi-node topology were unavailable, so
the original distributed workload was not claimed as reproduced.
