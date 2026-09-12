# Independent review of PR 3079

Reviewed candidate: https://github.com/amdpilot-org/sglang/pull/3079 at exact commit `8c5412debaf308d3c7ffc28a84727f324955a689`.

Upstream issue: https://github.com/sgl-project/sglang/issues/19087

Mirror issue: https://github.com/amdpilot-org/sglang/issues/3106

## Recommendation

`request_changes`. The candidate's submitted unit tests pass, and its source adds useful startup timing phases, but the actual worker entry is broken at runtime. `run_scheduler_process()` calls `get_startup_profiler()` without importing it. An independent test that executes the real function with only heavyweight scheduler/platform operations isolated fails at `gpu_worker.py:1601` with `NameError` before scheduler construction. The call is unconditional, so default-disabled launches are affected too.

The original issue is therefore not fully resolved by this exact commit. This is more than test-only hardening, but it remains an unshippable partial implementation.

## Additional coverage findings

- The parent `launch_server.prepare_processes` phase only surrounds the first master-to-slave pipe loop. Result pipes, scheduler pipes, `Process` object construction, and pipe cleanup remain outside that named preparation phase. `launch_server.total` does cover their wall time, but does not break it down.
- `SGLANG_DIFFUSION_STARTUP_BEGIN` is set inside `sglang.cli.generate.generate()`, after that module and its top-level dependencies have already imported. Thus `cli_and_import_preparation` cannot measure the complete CLI/import preparation interval named by the candidate.
- Parent and worker measurements are emitted as separate per-process summaries rather than aggregated into one timing tree. This can still provide useful evidence, but it does not substantiate the candidate prose's broad aggregation claim.

## Environment and architecture

The prepared interpreter was `/tmp/amdpilot-repo-j-d9b86da3600b/venv/bin/python`; import inspection confirmed both candidate modules came from `/job/repo/python`, not an installed wheel. The node exposes an AMD GPU through ROCm, but Qwen-Image weights were not present in this job's private runtime. No model/GPU run was claimed: the deterministic worker-entry failure occurs before model construction and does not require unavailable weights. The candidate changes Python only, so no native rebuild was applicable. Multi-GPU, CUDA, XPU, MPS, NPU, and disaggregated behavior remain unverified.

Raw commands and output are retained in this report directory. The candidate checkout was detached only for review and the repository was returned to `amdpilot/j-d9b86da3600b` before this report was committed.
