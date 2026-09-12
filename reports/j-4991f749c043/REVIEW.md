# Independent review of PR 2814 at `a919f1c5`

Recommendation: **request changes**. The candidate is a partial implementation and does not fully resolve the original issue.

The submitted focused suite passes (15 passed, 2 hardware skips), the candidate source imports were confirmed from `/job/repo/python`, and the changed embedded C++ allocator was rebuilt and loaded from `/tmp/symm_allocator/nccl_allocator.so`. Its new window-accessor exports were present, and a real allocation on the assigned AMD Instinct MI350X matched an independent CPU reference.

The candidate nevertheless introduces a blocking compatibility regression. `pynccl_wrapper.py` already had an `ncclConfig_t` with the `create()` initializer used by `PyNcclCommunicator` and `NCCLLibrary.ncclCommInitRankConfig`. The candidate adds a second class of the same name, shadowing the first and dropping `create()`. Against the real pinned RCCL, the supported default-config call now fails:

```text
AttributeError: type object 'ncclConfig_t' has no attribute 'create'
```

This affects the established symmetric-memory initialization path, not merely an unused helper. The candidate tests inspect the replacement struct but never exercise either pre-existing `create()` call site.

The base failure was independently reproduced: the recorded base has window registration but no RMA module, RMA function table, `put_signal`, or `nccl_mem_alloc` surface. The candidate meaningfully implements those pieces, so this is more than test-only hardening. It remains a partial fix because of the regression and because the central two-rank RMA success/fallback behavior could not be verified here.

Environment limits: one AMD Instinct MI350X (`gfx950`), Torch 2.11.0+rocm7.2, and RCCL 2.27.7. RCCL lacks all five RMA symbols, so no actual `put_signal`/`wait_signal`, NCCL >=2.30, NVLink, PCIe-only NULL-handle, or vGPU path ran. Raw evidence is retained at `/tmp/amdpilot-repo-j-4991f749c043/review-evidence` and detailed in `result.json`.
