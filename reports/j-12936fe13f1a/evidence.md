# Issue-specific evidence

At base `358c163250ad3b1f62939b01ce1314a0a31a0365`, the focused regression failed in three issue-specific ways:

- `TARGET_VERIFY` incremented `profiler_prefill_ct` instead of `profiler_decode_ct`.
- `DRAFT_EXTEND_V2` raised `RuntimeError: unsupported profile stage`.
- A mock exporter waiting five seconds kept `_stop_profile()` on its calling thread until the wait timed out.

After the change, the focused and adjacent suite reported:

```text
....ssssssssss                                                         [100%]
4 passed, 10 skipped, 17 warnings, 2 subtests passed in 8.79s
```

The assigned single GPU identified itself as `AMD Instinct MI355X`. A real profiler run through `SchedulerProfilerManager` produced:

```text
max_abs_error 3.0517578125e-05
trace_files ['gfx950-TP-0.trace.json.gz']
trace_sizes [42211]
```

Raw host-retained evidence is under `/tmp/amdpilot-repo-j-12936fe13f1a/evidence/`, and the generated trace is under `/tmp/amdpilot-repo-j-12936fe13f1a/gpu-profile/`.

This evidence does not represent a TP8/B200/GLM-5.2 serving reproduction. It establishes the faulty control flow, the scheduler-facing blocking behavior with a deterministic exporter, the corrected boundaries, and a real single-rank gfx950 collection/export path.
