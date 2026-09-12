# Investigation notes

- Prepared source: `/job/repo` at base `358c163250ad3b1f62939b01ce1314a0a31a0365`.
- Prepared interpreter: `/tmp/amdpilot-repo-j-132bdab73f8f/venv/bin/python`.
- Runtime/cache root: `/tmp/amdpilot-repo-j-132bdab73f8f`.
- Native rebuild: not applicable; no native source changed.
- Upstream issue was still open when inspected: https://github.com/sgl-project/sglang/issues/33698
- Mirror issue: https://github.com/amdpilot-org/sglang/issues/1746
- Related upstream PR inspected before implementation: https://github.com/sgl-project/sglang/pull/33701 (open at commit `901903d6ca72c4fcebd2e7ece8fb520ced44e46d`). Its review history identified the same two lifecycle constraints covered here: preserve unpaused RWLock writer preference and publish successful update metadata before propagating cancellation.

The base implementation stored one shared `model_update_result`, `model_update_tmp`, and `model_update_expected_workers` tuple of state. Paused calls bypassed the inference writer lock, allowing a second call to replace the first call's future before the first untagged scheduler response arrived. The failing-before log records the first caller remaining pending and a cancelled caller releasing ownership early.

The correction serializes ownership of those fields with `model_update_operation_lock`. The scheduler task is shielded from caller cancellation and drained before the operation lock is released. Paused operations retain `is_pause_cond` until completion; unpaused operations acquire the existing writer lock before the operation lock so already-requested updates keep preference over later inference readers.
