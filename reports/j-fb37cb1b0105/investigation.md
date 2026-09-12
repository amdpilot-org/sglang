# Investigation evidence

- Prepared checkout: `/job/repo`
- Prepared interpreter: `/tmp/amdpilot-repo-j-fb37cb1b0105/venv/bin/python`
- Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`
- Affected source: `python/sglang/srt/function_call/base_format_detector.py`
- Regression tests: `test/registered/unit/function_call/test_unknown_tool_name.py` and `test/registered/unit/function_call/test_function_call_parser.py`

The open source issue and mirror issue had no comments or linked resolution at inspection time. Upstream PR https://github.com/sgl-project/sglang/pull/23453 is related but primarily implements opt-in forwarding of unknown tools. Its later commit `1101b20a82a4787fdd3ca9a4bc3a643606662b8d` independently preserves the suffix after a complete unknown object, corroborating the buffer-loss diagnosis. It does not by itself handle the reported character-at-a-time unknown-first JSON-array case, which requires retaining partial unknown JSON until its boundary is known and remembering that the parallel sequence has started without consuming a client-visible tool index.

Raw before/after reproduction, failing-before regression, complete parser-suite output, and pre-commit output are retained under `reports/j-fb37cb1b0105/raw/`.
