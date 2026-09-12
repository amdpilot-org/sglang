# LMCache MP async-store correction

Candidate: https://github.com/amdpilot-org/sglang/pull/3491 at
`2ab32e15eddf8f8ba765a382dca23d475f0c6fc3`

Independent review: https://github.com/amdpilot-org/sglang/pull/3508

The candidate correctly made request-finish MP stores asynchronous and retained
radix locks until future reconciliation. Its remaining counterexample reproduced
at the exact commit: `evictable_size()` did not inspect completed futures, so a
locked store could suppress the capacity that would otherwise lead to `evict()`.

The consolidated correction polls each MP future once from `evictable_size()`.
Completed stores are settled and unlocked before the scheduler sees capacity;
unfinished stores return immediately and remain pinned. The existing blocking
eviction/reset/shutdown reconciliation remains the safety backstop. Query errors
also remain pinned because they do not prove that the cross-process copy is done.

Raw evidence:

- `raw/candidate_adversarial_before.txt`: exact candidate, 1 failed / 2 passed.
- `raw/pytest_async_store_final.txt`: corrected focused suite, 11 passed.
- `raw/review_adversarial_final.txt`: corrected review suite, 3 passed.
- `raw/environment_probe.txt`: frozen Torch/ROCm and missing LMCache evidence.
- `raw/py_compile_final.txt` and `raw/git_diff_check_final.txt`: static checks.

No real LMCache daemon test was possible. The installed interpreter has no
LMCache package, although Torch sees the prepared MI350X and exposes the ROCm
IPC-event API. Consequently this report makes no GPU execution claim.
