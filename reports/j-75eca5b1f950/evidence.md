# Investigation evidence

Base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

## Related-fix review

- Upstream PR #38869 already covers the issue's Pythonic-detector bracket-counting item, so that candidate was not duplicated.
- Upstream PRs #34925 and #35625 cover Step3 and broader streaming-parity work.
- Searches for Llama32 string-value corruption found no corresponding open fix. PR #32088 concerns the separate trailing-text off-by-one item.

## Failing-before reproduction

The production `Llama32Detector.parse_streaming_increment` was called with the required `Tool` object and these complete decoded outputs:

```text
<|python_tag|>{"name":"echo","arguments":{"text":"literal 'x': value"}}
<|python_tag|>{"name":"echo","arguments":{"text":"literal: 'x' value"}}
```

For both inputs the result contained zero calls. The retained buffers showed the corruption directly:

```text
<|python_tag|>{"name":"echo","arguments":{"text":"literal "x": value"}}
<|python_tag|>{"name":"echo","arguments":{"text":"literal: "x" value"}}
```

Thus the compatibility regex was applied inside a valid JSON string and produced invalid JSON before the base streaming parser saw it.

## Passing-after evidence

```text
$ PYTHONPATH=python /tmp/amdpilot-repo-j-75eca5b1f950/venv/bin/python -m pytest -q test/registered/unit/function_call/test_function_call_parser.py -k 'TestLlama32Detector'
..........                                                               [100%]
10 passed, 236 deselected, 1 warning in 4.13s

$ PYTHONPATH=python /tmp/amdpilot-repo-j-75eca5b1f950/venv/bin/python -m pytest -q test/registered/unit/function_call/test_function_call_parser.py
........................................................................ [ 29%]
.................................................................... [ 56%]
........................................................................ [ 86%]
..................................                                       [100%]
246 passed, 1 warning, 4 subtests passed in 8.02s
```

The warning is the repository's existing unknown `asyncio_mode` pytest configuration warning.

Source under test: `python/sglang/srt/function_call/llama32_detector.py`

Regression tests: `test/registered/unit/function_call/test_function_call_parser.py`
