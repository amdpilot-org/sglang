# Investigation evidence

- Prepared base: `358c163250ad3b1f62939b01ce1314a0a31a0365`
- Source under test: `python/sglang/srt/managers/scheduler_components/metrics_reporter.py`
- Regression: `test/registered/unit/observability/test_forward_pass_metrics.py`
- Failing-before output: `failing-before.log`
- Passing-after output: `passing-after.log`

The source issue remains open. Its linked upstream PR,
https://github.com/sgl-project/sglang/pull/31319, is also open and proposes the
same `req.seqlen` fallback, but it was not present in the prepared `main` base.
The local regression independently reproduced the reported `TypeError` through
`_emit_forward_pass_metrics()` before the implementation change.

This validation covers scheduler-side metric construction only. It does not
claim reproduction of the original GLM-5.1 architecture, 8x B200 deployment,
NSA/TRTLLM backend, DP collectives, or the subsequent Gloo failure cascade.
