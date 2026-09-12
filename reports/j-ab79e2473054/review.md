# Independent review of amdpilot-org/sglang PR 1966

Candidate reviewed: `a8b9c0fe45a4cc298d8ed84e52f8e956d5ae0f88`

Upstream issue: https://github.com/sgl-project/sglang/issues/33186

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2006

Recommendation: **request changes**. The candidate is a real but partial fix.

## Findings

The recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365` reproduced both examples from the issue: text supplied in an increment after a completed call remained buffered, and a split `<tool_call>` marker leaked as content and prevented call recognition.

At the exact candidate commit, its four focused tests pass (including all ten internal splits of `<tool_call>`). Independent direct-detector cases confirm that the two reported two-increment examples are corrected, and one-character chunking produces the call plus surrounding text without leaking markup.

However, normal text is still silently dropped when the final generated increment contains a complete tool call followed by text. For `CALL + "After."`, `parse_stream_chunk` emits the call and stores `"After."` in `MiMoDetector._buffer`. The serving path invokes `FunctionCallParser.parse_stream_end()`, but `MiMoDetector` does not override `BaseFormatDetector.finish()`, whose default returns an empty result. The suffix remains buffered after finalization and is never emitted. A final increment containing two calls likewise emits only the first and strands the second call and following text.

This falls within the original contract: arbitrary streaming chunk boundaries must not change the accumulated calls/content, and text after the last call must not be lost. The candidate tests cover trailing text only when it arrives in a later increment, which gives the parser another invocation and misses the final-increment boundary.

The one-shot MiMo parser on this base independently omits text after tool-call blocks, so exact one-shot/streaming normal-text equivalence cannot be established without addressing that pre-existing behavior. This does not excuse the serving loss demonstrated through the streaming parser's explicit end hook.

## Environment and scope

The prepared checkout was initially clean on `amdpilot/j-ab79e2473054` at the recorded base. At review time, fetched `origin/main` was `a207786205bff0919eb2c8c9126c67f302ccff34`, so it differed from the recorded base. Tests used `/tmp/amdpilot-repo-j-ab79e2473054/venv/bin/python`; imports resolved `sglang` and `mimo_detector.py` from `/job/repo/python`, while Torch resolved from `/opt/venv` (`2.11.0+rocm7.2`, HIP 7.2).

The candidate changes only Python detector code and tests; no native source changed, so no native rebuild was applicable. This deterministic parser review used no GPU. No MiMo weights or HTTP model-serving run was performed; model semantics, tokenizer delivery, and distributed execution remain outside the evidence. The public serving parser and its end-of-stream hook were exercised directly, which is sufficient to demonstrate the remaining parser/transport loss without weights.
