# Investigation evidence

- Prepared checkout: `/job/repo`
- Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`
- Interpreter: `/tmp/amdpilot-repo-j-74c07e9212b8/venv/bin/python`
- Changed source: `python/sglang/srt/function_call/llama32_detector.py`
- Focused regression: `test/registered/unit/function_call/test_llama32_detector.py`
- Original issue: https://github.com/sgl-project/sglang/issues/35562
- Mirror issue: https://github.com/amdpilot-org/sglang/issues/1356
- Related upstream change inspected: https://github.com/sgl-project/sglang/pull/36671 (closed, unmerged)

The before-fix direct reproduction is retained in `raw/reproduction-before.txt`.
The regression was added before the source change and its four failing subtests
are retained in `raw/regression-before.txt`. Passing output from the same test
file is in `raw/regression-after.txt`, and direct public-registry assertions are
in `raw/reproduction-after.txt`.

This is a deterministic parser-only defect. GPU execution, model weights, an
HTTP server, native compilation, and distributed execution are not involved,
and none are claimed as validation.
