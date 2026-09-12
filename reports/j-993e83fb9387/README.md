# Independent review of PR 2155

Upstream issue: https://github.com/sgl-project/sglang/issues/33385

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2189

Candidate: https://github.com/amdpilot-org/sglang/pull/2155 at `19e9c3719ed7600c637fafbe14890b87113666a3`

The prepared base `358c163250ad3b1f62939b01ce1314a0a31a0365` reproduces both relevant failures in the candidate's regression: decode retraction with CPU-tensor backup still calls `offload_kv_cache` when decode KV offload is disabled, and an unsupported `get_cpu_copy()` raises `NotImplementedError` instead of selecting recomputation.

At the exact candidate revision, all four focused regression cases pass. The candidate respects the disabled-offload flag, maps only `NotImplementedError` to the existing PD rebootstrap path, preserves host-pool exhaustion as a distinct failure, and continues to propagate unrelated copy failures. The supported MHA target/draft exact-value GPU backup/restore tests also pass.

The candidate is not cleanly acceptable at this commit. Its change to `_add_request_to_queue` always forwards `is_rebootstrap=False`; the existing decode priority-queue contract test expects the prior call shape and fails. The broader focused set produced 1 failure, 42 passes, and 5 passing subtests. This appears to require candidate test/compatibility hardening rather than a different issue fix, but the reviewed commit itself is red.

The original tp8/dp8 DeepSeek-V4 speculative HiSparse serving workload was not available. The assigned single AMD GPU and prepared environment cannot establish distributed race timing, DeepSeek-V4 semantics, or full serving behavior. No native source changed, so no native rebuild was applicable. Source imports were confirmed from `/job/repo/python/sglang` at the candidate revision.
