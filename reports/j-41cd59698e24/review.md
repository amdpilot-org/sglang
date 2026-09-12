# Independent review of PR 2952 at 34d8613

Upstream issue: https://github.com/sgl-project/sglang/issues/32200

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2984

Candidate: https://github.com/amdpilot-org/sglang/pull/2952 at `34d861306b4fdce616d2260d9d9f52655d6ef145`

Recommendation: **request changes**. This is a partial fix, not a full resolution of the original issue.

The recorded base lacks the requested RMA bindings, communicator, and operations. Against the preceding candidate `a919f1c5af0c12fe6d9c94af9f70633b3780996f`, I independently reproduced both reported compatibility failures: the default configured-init call and symmetric-memory communicator construction raise `AttributeError` because the replacement `ncclConfig_t` shadows the established class and has no `create()` method.

The reviewed commit correctly consolidates the struct, preserves `create()`, and adds direct regression tests for both call sites. Its focused suite reports 18 passed and 2 hardware tests skipped.

However, an independent native rebuild and adversarial probe found a remaining original-contract failure. `get_nccl_mem_pool()` declares only `_allocator`, `_mem_pool`, `_cur_device`, and `_register_func` as globals. Its assignments to `_get_windows_func` and `_clear_windows_func` are consequently local. The rebuilt `/tmp/symm_allocator/nccl_allocator.so` exports both new functions, yet after loading it the module globals are still `None`. `_collect_windows_for_comm()` therefore always returns `[]`, and `SymmetricMemoryContext.windows` cannot surface any registered handles. Without those handles, the proposed RMA communicator cannot consume the allocator's registered windows as required.

Environment limitation: the node has one AMD Instinct MI355X with ROCm 7.2. The changed allocator C++ was rebuilt and loaded with `-lrccl`, but the two-rank NVIDIA NCCL 2.30/NVLink RMA handshake cannot run here. This limitation is separate from the deterministic Python scoping defect above.

Raw revision-independent logs and captured diffs are retained under `/job/review-evidence-j-41cd59698e24/`.
