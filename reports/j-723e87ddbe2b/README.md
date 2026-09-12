# Diffusion startup profiler correction generation 2

Reviewed candidate: https://github.com/amdpilot-org/sglang/pull/3079 at exact commit `8c5412debaf308d3c7ffc28a84727f324955a689`.

Independent review: https://github.com/amdpilot-org/sglang/pull/3157

Upstream issue: https://github.com/sgl-project/sglang/issues/19087

Mirror issue: https://github.com/amdpilot-org/sglang/issues/3158

## Reproduction and correction

The candidate's real `run_scheduler_process()` target was executed with only platform, scheduler, and logging operations isolated. It failed before scheduler construction with `NameError: name 'get_startup_profiler' is not defined`, including with startup profiling disabled. The raw traceback is retained in `candidate-worker-entry.log`.

Source inspection also confirmed the other review findings: the CLI clock began inside `generate()`, the named preparation phase covered only task pipes, and workers logged separate process-local summaries.

The correction preserves the candidate's model/component instrumentation and:

- imports and exercises `get_startup_profiler` in the real worker target;
- starts the console-path clock in `sglang.cli.main`, before importing `sglang.cli.generate`, with a direct-call fallback at the beginning of the generate module;
- records task pipes, result pipes, scheduler pipes/process construction, each process start, parent pipe cleanup, and ready waiting;
- transfers each worker's pickle-safe timing snapshot through the existing ready pipe and attaches it to the parent's single `launch_server.total` tree.

## Validation

```text
PYTHONPATH=/job/repo/python /tmp/amdpilot-repo-j-723e87ddbe2b/venv/bin/python -m pytest -q python/sglang/multimodal_gen/test/unit/test_startup_profiler.py python/sglang/multimodal_gen/test/unit/test_launch_server_shutdown.py
14 passed, 16 warnings in 1.12s
```

`py_compile` on all changed runtime/CLI profiler modules and `git diff --check` also passed. Ruff was unavailable in the prepared interpreter.

## Limitations

No Qwen-Image weights were available in the prepared private runtime, so no new model launch or GPU execution is claimed. This correction addresses deterministic Python control-flow and accounting defects that reproduce before model construction. Multi-node, multi-GPU, non-ROCm, and disaggregated launches remain unverified. No native code changed, so a native rebuild was not applicable.
