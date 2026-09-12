# Independent review of PR 2150

Candidate: https://github.com/amdpilot-org/sglang/pull/2150  
Exact commit: `ad317f85a69d0af572d618d10c52152dee58000a`  
Upstream issue: https://github.com/sgl-project/sglang/issues/33186  
Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2185

## Recommendation

Accept. The candidate fully resolves the original parser contract in the tested pure-Python scope.

On the recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`, the public streaming parser stranded trailing text after a completed call, processed only the first of two complete calls in one increment, and leaked a split `<tool_call>` marker as normal content. The exact candidate emits trailing and inter-call text, drains all complete calls, preserves sequential tool indexes, handles marker splits, and makes one-shot normal text include text between and after calls.

The candidate's focused suite passed: 9 tests, 204 subtests. An independent differential harness passed 1,580 chunkings across trailing text, multiple and adjacent calls, every single-character stream, randomized chunk sizes, unknown tools, malformed blocks, partial markers at EOF, ordinary angle-bracket text, and newline-bearing suffixes. Streaming output and accumulated calls matched `detect_and_parse`, and the detector buffer was empty after end-of-stream finalization.

## Environment and limitations

The loaded candidate source was `/job/repo/python/sglang/srt/function_call/mimo_detector.py`, verified with `inspect.getfile`. This change touches only Python parser code, so no native rebuild was applicable. No GPU was used: the defect and regression are deterministic CPU-only parsing behavior. No MiMo weights, tokenizer, HTTP server, model semantic accuracy, distributed serving, or architecture-specific execution was tested. Those limitations do not block verification of the original issue's stated parser contract.

Raw command output is retained under `raw/`.
