# Independent review of candidate PR 3491

Reviewed exact candidate commit `2ab32e15eddf8f8ba765a382dca23d475f0c6fc3`
against recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`.

Upstream issue: https://github.com/sgl-project/sglang/issues/31842

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/3484

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/3494

Candidate PR: https://github.com/amdpilot-org/sglang/pull/3491

## Recommendation

`request_changes`. The candidate implements a genuinely asynchronous submit,
keeps the node locked through its real future, and correctly drains cleanup on
`evict()`, `reset()`, and graceful shutdown. Its focused suite passes at the
exact commit, and the same suite fails on the recorded base.

It does not fully resolve slot reuse. Completed MP futures are only reconciled
inside `evict()`, `reset()`, or shutdown. Locked nodes are excluded from
`RadixCache.evictable_size()`. When the outstanding stores hold the capacity
needed to form another batch, the scheduler can observe no evictable capacity
and never reach the allocation/eviction path. The completed future then remains
unobserved and its node remains pinned indefinitely. The independent
`test_completed_store_is_not_reaped_at_scheduler_budget_boundary` reproduces
this counterexample at the candidate commit.

The related current upstream PR 32455 independently describes this circular
budget/eviction failure and adds a non-blocking reap at `evictable_size()`.
That upstream PR is corroborating context, not proof; the retained adversarial
test directly demonstrates the missing candidate behavior.

## Evidence

- Recorded base: candidate regression suite produced 6 failures and 1 pass;
  the base uses blocking `store_kv`, immediately unlocks, clears the marker,
  and ends the session.
- Exact candidate: its 7 focused tests pass and Python compilation plus
  `git diff --check` pass.
- Independent adversarial suite: 1 failure and 2 passes. The failing case is
  completed-future reaping at the scheduler budget boundary. The passing cases
  show all entries are drained after an earlier future fails and IP sync errors
  do not prevent MP cleanup.
- LMCache PR 4152 is merged at `f7fffc186687d710a711ba6e8f48f0af6716fb8d`.
  Its `store_kv_async` returns a CUDA messaging future whose `result()` waits
  for both the raw daemon response and imported event synchronization, matching
  the candidate's completion primitive.
- The mandated interpreter imports SGLang from `/job/repo/python`. No LMCache
  distribution is installed (`lmcache_spec None`), so real connector/daemon
  integration could not be executed.
- Hardware visibility: one AMD Instinct MI350X under Torch 2.11.0+rocm7.2 / HIP
  7.2.26015. No GPU workload was run because the required LMCache dependency
  and daemon were absent. ROCm exposes `Event.ipc_handle` and
  `Event.from_ipc_handle`, but this alone does not qualify the full integration.
- No native source changed, and repository metadata provides no native rebuild
  target; `native_rebuilt` is therefore false and not applicable.

Raw command output and the independent test are retained under `raw/`.

