# Diffusion startup profiling correction

This correction preserves the worker/model-loading instrumentation from
https://github.com/amdpilot-org/sglang/pull/2921 and addresses the concrete
counterexamples reported by https://github.com/amdpilot-org/sglang/pull/3010.

## Failing-before evidence

- At candidate commit `8205569c20da3c3019476dcb3d7e8ded26514532`, a fresh
  `gpu_worker` import took 9.586 seconds in this prepared environment. It occurs
  before `run_scheduler_process`, so the candidate tree could not include it.
- The retained candidate Qwen log runs from 18:23:11 until profile emission at
  18:23:42, while `init_scheduler` reports 20.936 seconds.
- Two calls to `log_startup_summary()` produced `log_calls=2`.
- The parent `launch_server` path had no phases around pipe/process preparation,
  `Process.start()`, or the blocking ready wait.

Raw reproduction output is retained in
`/job/review-evidence-j-18df02305d79/`.

## Correction

- Capture CLI/import preparation for the `sglang generate` Qwen example.
- Report worker process start plus spawned-interpreter imports before scheduler
  initialization.
- Measure parent process preparation, each process start, total ready wait, and
  total `launch_server` duration.
- Make summary logging idempotent per process.

The focused after suite passes 11 tests. Qwen-Image weights were not present in
the prepared private runtime, so the retained parent GPU run was not
speculatively replaced with a different architecture. No native source changed.
