# Independent review of PR 1529 at 3e5ecbb8089e8540d803745cff5b19ea17453c7e

The image-prepared checkout was exactly the requested recorded base,
`358c163250ad3b1f62939b01ce1314a0a31a0365`, before review. The candidate was
checked out detached at the exact requested commit, and the checkout was returned
to `amdpilot/j-793cff37cf4d` before this report was committed.

## Findings

The candidate is a partial functional fix, but is not safe to accept as written.

1. The stage-classification correction is effective. On the base,
   `TARGET_VERIFY` increments the prefill counter, `DRAFT_EXTEND_V2` is rejected,
   and a blocked trace export blocks `_stop_profile()`. The candidate's focused
   regression passes all four tests and both speculative-mode subtests.
2. Moving export to a background thread makes the mocked scheduler-facing stop
   return promptly, and an independent single-GPU ROCm run exported a nonempty
   43,006-byte gzip trace after a numerically checked matrix multiplication.
3. The background thread also calls `torch.distributed.barrier()` on
   `dp_tp_cpu_group`, the same process group used by normal scheduler collectives.
   An independent two-rank Gloo adversarial run with staggered export times and a
   concurrent main-thread `all_reduce` hung until the 25-second outer timeout
   (exit 124). This demonstrates a collective-ordering race in precisely the
   asynchronous design that would be used in the reported TP8 configuration.
   The candidate's world-size-one test cannot expose this problem.
4. The committed regression hard-codes
   `/tmp/amdpilot-repo-j-12936fe13f1a` as the parent of a temporary directory.
   With that campaign-private directory absent, the test fails with
   `FileNotFoundError`, so it is not portable to normal CI or another checkout.

The changed files are Python-only. No C++, HIP, CUDA, FlyDSL, or other native
source changed, so no native rebuild was applicable. Import inspection at the
candidate confirmed that `sglang`, `profiler_manager.py`, and `profile_utils.py`
were loaded from `/job/repo/python`, while Torch came from the prepared
`/opt/venv` installation (`2.11.0+rocm7.2`, HIP 7.2).

## Environment and scope

The assigned device reported `AMD Instinct MI350X`, capability `(9, 5)` (gfx950),
and only one GPU was available. The review therefore does not reproduce the
reported B200 x8, TP8, GLM-5.2, MTP workload, 242 MB trace, 25-second duration, or
end-to-end TTFT increase. The local two-process Gloo test is an independent
control-flow/concurrency counterexample, not a substitute for that unavailable
model workload. No model weights were used. `SGLANG_PROFILE_V2` still performs
its own synchronous export; the candidate corrects its stage mapping but does not
make that exporter asynchronous.

Raw logs and the standalone adversarial scripts are retained outside the checkout
under `/job/review-evidence/`. The generated GPU trace is retained under
`/tmp/amdpilot-repo-j-793cff37cf4d/gpu-profile/`.
