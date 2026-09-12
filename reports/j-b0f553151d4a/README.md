# Independent review of PR 2568

Candidate reviewed: `9fcf66b7818e6bed7c2d510fbbb526c1c56a0aed`

Upstream issue: https://github.com/sgl-project/sglang/issues/6357

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/2512

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2572

## Recommendation

Accept. The candidate fully resolves the original metric-availability contract.
It adds the missing scheduler state, registers and emits the Prometheus gauge,
and refreshes its value in both prefill and decode reporting. The implementation
uses the existing monotonic `wait_queue_entry_time` clock and handles empty,
uninitialized, future, NaN, and infinite timestamp inputs without producing a
negative or non-finite gauge.

This is a full original-issue fix, not merely test hardening: on the recorded
base, an isolated Prometheus registry probe exited 3 with
`has_metric: False`; at the exact candidate it exited 0 and exposed the metric.
An independent candidate probe emitted this exact sample after logging a value
of 2.5:

```
sglang:avg_request_queue_latency{model_name="review",moe_ep_rank="0"} 2.5
```

The candidate's focused regression passed 4 tests. The existing Ray metrics
wrapper suite passed 33 tests and 4 subtests. Independent adversarial cases
also passed, including exclusion of negative/zero/NaN timestamps, mixed
past/future timestamps, and an infinite future timestamp. `compileall` and
`git diff --check` passed.

## Source and native-path verification

Both probes imported `sglang` from
`/job/repo/python/sglang/__init__.py`. The candidate changes only Python source,
tests, and report artifacts; it changes no C++, CUDA, HIP, or other native
source. The prepared environment declares `native: null` and `wheel: null`, so
no native rebuild was applicable.

## Architecture and environment limitations

The independent review did not launch a model server or execute GPU kernels.
The assigned device was observed idle and identified as one AMD Instinct MI350X
(`gfx950`, ROCm 7.2), which cannot reproduce the report's eight-GPU NVIDIA,
TP=8, DP-attention deployment. No production model weights were available.

The candidate includes retained evidence of a single-gfx950 tiny-Llama serving
run with two HTTP 200 responses and a concurrent positive queue-latency sample.
That evidence was inspected but was not treated as proof of NVIDIA, TP=8,
DP-attention, distributed, model-semantic, or production-load behavior. Those
dimensions remain untested, but they do not create a remaining counterexample
to the architecture-independent Prometheus registration and scheduler
calculation fixed here.

Raw review outputs and the immutable candidate patch were preserved outside the
checkout at `/job/review-evidence-j-b0f553151d4a/` while revisions were
switched. No candidate source was copied into this branch.

## Reproduction commands

```bash
# At recorded base 358c163... (expected exit 3, has_metric: False)
/tmp/amdpilot-repo-j-b0f553151d4a/venv/bin/python \
  /job/review-evidence-j-b0f553151d4a/base_metric_probe.py

# At exact candidate 9fcf66b... (expected exit 0, has_metric: True)
/tmp/amdpilot-repo-j-b0f553151d4a/venv/bin/python \
  /job/review-evidence-j-b0f553151d4a/base_metric_probe.py

/tmp/amdpilot-repo-j-b0f553151d4a/venv/bin/python -m pytest -q \
  test/registered/unit/observability/test_queue_latency_metrics.py

/tmp/amdpilot-repo-j-b0f553151d4a/venv/bin/python -m pytest -q \
  test/registered/unit/observability/test_ray_wrappers.py

/tmp/amdpilot-repo-j-b0f553151d4a/venv/bin/python \
  /job/review-evidence-j-b0f553151d4a/candidate_adversarial_probe.py
```
