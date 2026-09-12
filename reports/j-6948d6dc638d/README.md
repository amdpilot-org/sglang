# Investigation of sglang#33687

The reported hardcoded `/tmp/shm_wr_lock.lock` defect is already absent at the
prepared base, `358c163250ad3b1f62939b01ce1314a0a31a0365`.

Upstream PR [#33949](https://github.com/sgl-project/sglang/pull/33949), merged as
commit `443b62db57a964cf7261210721070be2da654fd5`, replaced the old POSIX
shared-memory consumption counter and global file lock with generation-tagged
GPU control words ordered on producer and consumer CUDA streams. The former
module is now only a compatibility re-export, and neither the active CUDA IPC
transport nor its shared stream-ordered pool imports `fcntl` or refers to
`shm_wr_lock`.

The reporter's proposed per-user-path PR
[#33690](https://github.com/sgl-project/sglang/pull/33690) remains open, but its
patch targets the pre-#33949 implementation and is obsolete on this base.

## Evidence

- `rg -n "SHM_LOCK_FILE|shm_wr_lock|fcntl|flock"` over the active transport and
  compatibility module returns no matches.
- `CudaIpcTensorTransportProxy.acknowledge_consumption` now calls
  `_acknowledge_on_stream`, which writes the lease generation to a per-consumer
  GPU control word through `cuStreamWriteValue32`; it performs no filesystem
  operation.
- The focused CPU transport suites pass: 40 tests and 2 subtests. These cover
  pool budgeting, transport setup and teardown, error cleanup, packed features,
  and independent invalid/disabled boundary behavior.
- A real one-GPU cross-process CUDA IPC test was attempted on the assigned AMD
  Instinct MI350X (`gfx950`). It cannot validate this NVIDIA CUDA-only path: the
  spawned producer stops at `ModuleNotFoundError: No module named 'cuda'` when
  the implementation requests CUDA driver bindings. No model or NVIDIA GPU was
  available, so no full serving/VLM claim is made.

No product-source correction was justified. This PR records the already-fixed
state and the validation boundary.
