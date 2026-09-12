# Independent review of amdpilot-org/sglang PR 2241

- Upstream issue: https://github.com/sgl-project/sglang/issues/31896
- Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/2177
- Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2278
- Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`
- Exact candidate: `83323d17b6bd11c450723f5efd3b766f4e556905`
- Recommendation: **accept**

## Finding

The candidate fully fixes the reported default-metrics exporter duplication at
the implementation point that causes it. On the recorded base,
`SchedulerMetricsCollector.init_new` marks every attention context-parallel
rank whose attention tensor-parallel rank is zero as a stats/export rank. The
candidate additionally requires attention CP rank zero. The submitted CP=8
regression therefore fails on the base with eight enabled exporters and passes
on the exact candidate with one.

This is a source fix, not test-only hardening. `num_running_reqs`, queue counts,
and the other scheduler metrics share `current_scheduler_metrics_enabled`, so
the changed rank predicate applies to the running and waiting gauges described
by the issue. The explicit `enable_metrics_for_all_schedulers` option still
enables every scheduler by design; the candidate only changes the default.

An independently authored matrix exercised CP sizes 1, 2, 3, and 8 crossed
with attention-TP sizes 1, 2, and 4. Every default topology selected exactly
one exporter, while every explicit all-schedulers topology enabled all ranks
and retained exactly one human-log leader. The complete observability unit
directory also passed.

## Source, native, and environment verification

The exact candidate was checked out detached for validation, then this review
branch was restored at the recorded base before the report was committed.
Python imported `sglang` and `metrics_collector.py` from `/job/repo/python`, not
from an unrelated installed copy. The candidate changes only Python and report
files; it changes no C++, FlyDSL, extension, or other native source, so no
native rebuild is applicable.

The interpreter was `/tmp/amdpilot-repo-j-cad949489e1a/venv/bin/python`, with
Torch `2.11.0+rocm7.2` and HIP `7.2.26015`. It saw one AMD Instinct MI350X
device (gfx950 family). This differs from the candidate report's MI355X device
label, but both inventories establish the material limitation: only one GPU
was available.

## Limitations

The original eight-H20 CP=8 HTTP/model-serving deployment could not be run on
one gfx950-family GPU, and no H20 hardware or model weights were supplied. No
full-model, multi-GPU, multi-node, transport, semantic-accuracy, or performance
claim is made. The deterministic test directly validates the actual scheduler
rank-selection and exporter cardinality responsible for the issue, but not an
end-to-end Prometheus scrape of the unavailable deployment.

Raw outputs are retained in `evidence/`.
