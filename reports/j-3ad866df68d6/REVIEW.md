# Consolidated correction for PR 982 after PR 1070

Candidate: https://github.com/amdpilot-org/sglang/pull/982 at `956321de134fde54ee75b71aad60be1546163183`.

Independent review: https://github.com/amdpilot-org/sglang/pull/1070.

Both review counterexamples reproduced at the exact candidate. An aggregate-fitting 30+20+20+20 GiB allocation was rejected because the 30 GiB rank was capped at an equal 22.5 GiB share. Two independently launched job contexts also each accepted 60 GiB from the same 90 GiB usable snapshot, oversubscribing the host by 30 GiB.

The correction retains coverage of all six HiCache allocation guards but replaces the process-group snapshot/equal-share decision with a host-local advisory lock. Each guard holds the lock across a fresh `psutil` reading and the allocation itself. This admits unequal requests according to the real remaining host budget, prevents co-located ranks from double-charging earlier allocations, and coordinates separately launched SGLang jobs that share the host and temporary directory.

The deterministic real-process regression uses the actual lock implementation. Four processes requesting 30+20+20+20 GiB all passed and consumed the 90 GiB fixture. In a second run, only one of two independent 60 GiB processes passed against 90 GiB usable. The focused and adjacent unit tests passed: 42 passed, 1 skipped, with 15 subtests passed.

Limitations: no actual 8xH200 GLM-5.2-W4AFP8 workload or model weights were available. The tests exercise CPU host accounting and real OS `flock`, not GPU kernels. One AMD Instinct MI355X gfx950 was visible but unused. The cross-job lock requires the jobs to see the same local temporary directory; isolated containers with distinct `/tmp` mounts or non-cooperating allocators remain outside this mechanism.
