# Correction generation 2: custom all-reduce concurrency

Upstream issue: https://github.com/sgl-project/sglang/issues/31117

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2614

Candidate parent PR: https://github.com/amdpilot-org/sglang/pull/2567 at exact
commit `7666e8a3ae6ddc1fef450f226d5b14ea009034e0`.

Independent review parent PR: https://github.com/amdpilot-org/sglang/pull/2612.

## Result

The exact candidate's eager serialization fixes are preserved. The review's
CUDA-graph counterexample was confirmed from source: capture returned without
changing guard state, and replay cannot invoke the Python guard. This correction
prevents both legacy and V2 custom all-reduce from being selected during active
graph capture, allowing the existing graph-safe collective fallback to be used.
Direct capture-time custom-AR calls now fail before enqueue instead of silently
recording an unsafe graph.

The static graph contract check fails against the exact candidate (exit 1) and
passes against this correction (exit 0). Twelve focused unit/capability tests
pass. On the assigned AMD Instinct MI350X (`gfx950`), the retained two-thread,
two-stream eager ordering probe produced `2`, equal to its CPU reference, and a
real HIP graph capture was rejected with unchanged guard state.

## Remaining limitation

Source inspection independently confirms that the JIT one-shot push poll and
legacy AOT barrier polls remain unbounded. No timeout was added: this assignment
provides one ROCm gfx950 GPU, not two CUDA A100 NVLink GPUs, so it cannot establish
a safe CUDA deadline, inspect the resulting CUDA ISA, or exercise timeout behavior
under the reported collision. A speculative iteration or clock limit could turn
ordinary rank skew into false device failures. The original two-rank CUDA
deadlock, green-context workload, concurrent replay of two CUDA graphs, and full
model serving remain unexecuted here.

No native source changed, so a native rebuild was not applicable.

## Evidence

- `raw/regression_before_candidate.txt` and `raw/regression_after.txt`: exact
  candidate fails and corrected source passes the graph-capture contract.
- `raw/focused_tests_final.txt`: 12 focused tests pass.
- `raw/gfx950_host_thread_probe.txt`: real-GPU eager ordering result.
- `raw/gfx950_capture_guard_probe.txt`: real-GPU graph capture rejection.
- `raw/native_unbounded_loops.txt`: remaining JIT and legacy unbounded waits.
- `raw/gpu_inventory.txt`: assigned hardware and ROCm runtime.
