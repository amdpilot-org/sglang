# Investigation evidence

- Upstream issue: https://github.com/sgl-project/sglang/issues/33901 (open when inspected)
- Mirror issue: https://github.com/amdpilot-org/sglang/issues/1714 (open when inspected)
- Prepared base: `358c163250ad3b1f62939b01ce1314a0a31a0365`
- Source path: `python/sglang/srt/function_call/glm47_moe_detector.py`
- Regression path: `test/registered/unit/function_call/test_function_call_parser.py`

The prepared source still returned immediately after `_finalize_tool_call()` even though that helper stored the unmatched suffix back in `_buffer`. Existing GLM-4.7 coverage split the second call across later increments, so it did not exercise a final delta containing two complete calls.

`failing-before.log` records the issue-specific baseline: index 0 emitted and the complete second call buffered. `passing-after.log` records both sequential indices and an empty buffer after the correction. The focused test and lint logs retain the remaining validation output.
