# Consolidated correction evidence

Prepared base: `358c163250ad3b1f62939b01ce1314a0a31a0365`.
Candidate reviewed: https://github.com/amdpilot-org/sglang/pull/1529 at
`3e5ecbb8089e8540d803745cff5b19ea17453c7e`.
Independent review: https://github.com/amdpilot-org/sglang/pull/1638.

## Failing before

All review counterexamples were checked against the exact candidate in the
detached worktree `/tmp/amdpilot-candidate-j-b0e05ade0242` before changing the
delivery branch.

* With `/tmp/amdpilot-repo-j-12936fe13f1a` absent, the candidate's
  `test_trace_export_does_not_block_stop` failed with `FileNotFoundError` while
  constructing its `TemporaryDirectory` (exit 1).
* A two-rank Gloo fixture staggered trace export while each scheduler/main
  thread issued `all_reduce` on the same group. The candidate timed out after
  15 seconds (exit 124), reproducing the collective-ordering hang.
* Source inspection and a blocking mocked exporter confirmed that
  `_ProfilerTorch.stop()` in `SGLANG_PROFILE_V2` called
  `export_chrome_trace()` and `barrier()` synchronously.

## Passing after

The default and v2 implementations now stop collection synchronously but write
the trace on tracked background threads. Export threads do not issue distributed
collectives. Optional merging coordinates completed files in the output
directory instead of using the scheduler process group.

```text
PYTHONPATH=python /tmp/amdpilot-repo-j-b0e05ade0242/venv/bin/python -m pytest -q \
  test/registered/unit/managers/test_profiler_manager_stage_stop.py \
  test/registered/unit/hardware_backend/mlx/test_metal_profiler.py
5 passed, 10 skipped, 17 warnings, 2 subtests passed
```

The retained `adversarial_collective.py` two-rank Gloo fixture completed under a
60-second outer timeout. Rank
0 and rank 1 both printed an all-reduce result of `3.0` (exit 0). A 30-second
limit was too tight for repeated fresh-process imports on this image even though
both ranks had completed; the 60-second result is the recorded correctness run.

The retained gfx950 script `gpu_profile_v2.py` executed a 256x256 GPU matrix
multiplication through the changed v2 profiler. It produced a 41,705-byte gzip
trace at `/tmp/amdpilot-repo-j-b0e05ade0242/gpu-profile-v2/`, the exporter
thread terminated, and GPU output differed from the CPU reference by at most
`3.0517578125e-05`.

## Limitations

Only one AMD Instinct MI350X/gfx950 GPU was assigned. There was no B200 x8,
GLM-5.2 weights, or TP8/MTP serving workload, so the reported 25-second export
duration and end-to-end TTFT increase were not reproduced. The two-process Gloo
test validates the process-group ordering defect, not multi-rank GPU profiling.
No native source changed, so no native rebuild was applicable.
