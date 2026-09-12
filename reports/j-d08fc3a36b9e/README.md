# Independent review of PR 2921 at `8205569c20da3c3019476dcb3d7e8ded26514532`

Recommendation: **request changes**. The candidate is a useful partial implementation, but it does not fully resolve the original request for a detailed breakdown of diffusion launch time.

## Findings

1. **The timing tree starts too late to account for the complete launch.** `startup_phase("init_scheduler")` begins inside `run_scheduler_process`, after the spawned worker has imported `gpu_worker.py` and its large dependency graph. The source even documents that "spawned workers import model dependencies before entering run_scheduler_process". An independent fresh interpreter import of that module took 11.383 seconds on the prepared node. The candidate's retained Qwen-Image log independently exposes the same gap: its first timestamp is 18:23:11, the profile is emitted at 18:23:42, but `init_scheduler` accounts for only 20.936 seconds. Thus roughly ten seconds of that observed launch is absent from the hierarchy. Parent-side CLI/import preparation, pipe/process creation, and wait-for-ready time are also unmeasured. This is material to the optimization question in the original issue.

2. **The implemented worker/model-load breakdown is credible.** At the exact candidate commit, all five candidate unit tests passed. Changed Python modules compiled and imported from `/job/repo/python`, not an installed copy. The retained Qwen-Image log contains a coherent hierarchy covering scheduler, distributed initialization, pipeline discovery/instantiation, configuration, named component loads, initialization, and stage creation.

3. **The documented "log once" property is not enforced by the profiler.** An independent adversarial call of `log_startup_summary()` twice produced two logger calls. The current production path calls it once, so this is secondary rather than the basis of the recommendation, but the API/docstring claim is stronger than the implementation.

## Base reproduction

On recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`, importing from the prepared source showed that `sglang.multimodal_gen.runtime.utils.startup_profiler` did not exist, and a source search found no startup-profile instrumentation. The prepared checkout matched the recorded base; there was no image/base discrepancy.

## Commands and evidence

- Base source/import check: `PYTHONPATH=/job/repo/python /tmp/amdpilot-repo-j-d08fc3a36b9e/venv/bin/python ...` (exit 0; profiler spec was `None`).
- Candidate regression: `PYTHONPATH=/job/repo/python /tmp/amdpilot-repo-j-d08fc3a36b9e/venv/bin/python -m pytest -q python/sglang/multimodal_gen/test/unit/test_startup_profiler.py` (exit 0; 5 passed).
- Candidate compile/import checks against `/job/repo/python` (exit 0).
- Independent fresh-process `gpu_worker` import timing (exit 0; 11.383 seconds).
- Independent repeated-summary adversarial case (exit 0; two invocations yielded two log calls).
- `git diff --check 358c163250ad3b1f62939b01ce1314a0a31a0365..8205569c20da3c3019476dcb3d7e8ded26514532` (exit 0).

Raw logs, fetched issue/PR JSON, the exact candidate patch, exit-code files, and SHA-256-addressable evidence remain outside the revision-switching checkout at `/job/review-evidence-j-d08fc3a36b9e/`.

## Environment and architecture limits

The node exposes one AMD Instinct MI355X (`gfx950`) with ROCm 7.2 and Torch 2.11.0+rocm7.2. No Qwen-Image weights were present in this review job's private runtime, so the candidate's retained GPU run could be inspected but not independently repeated without downloading a large model. This review did not claim independent GPU execution or semantic/numerical validation. The patch changes Python only; no C++/CUDA/HIP/FlyDSL/native source changed, so a native rebuild was not applicable. Multi-GPU, CUDA, XPU, MPS, NPU, and disaggregated paths remain unverified.
