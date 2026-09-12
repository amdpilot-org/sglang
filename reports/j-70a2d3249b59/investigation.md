# DeepSeek V4 mixed content/tool-call streaming investigation

Upstream issue: https://github.com/sgl-project/sglang/issues/34214

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1635

Base commit: `358c163250ad3b1f62939b01ce1314a0a31a0365`

## Finding

The prepared source already contains the relevant correction from merged upstream PR https://github.com/sgl-project/sglang/pull/34458. No duplicate source change was made.

The issue occurs when ordinary content and the opening DeepSeek V4 DSML tool marker arrive in the same parser increment. Before PR #34458, `DeepSeekV32Detector.parse_streaming_increment` returned an empty `normal_text` once it found an invoke block, discarding the ordinary-text prefix held in the same buffer. The current implementation computes `preamble` before the wrapper/invoke and returns it with parsed calls.

## Failing-before/passing-after replay

The replay loaded the detector source from PR #34458's base commit (`7fb6e61b953e6598aa6b0bffad4c9db0435e734e`) alongside the detector in this checkout. Its chunks mirror the issue excerpt, with the missing suffix and DSML call sharing the final increment.

- Before: `我来为您调用工具获取北京天气，同时为您呈现毛主席`
- Current: `我来为您调用工具获取北京天气，同时为您呈现毛主席的《沁园春·雪》`
- Both paths parsed `get_current_weather` with `{"location":"Beijing","unit":"celsius"}`.

An independent boundary split the opening marker between `tool_` and `calls>`. Before the fix all preamble content was lost; current source returned `完整前言`. Content-only and tool-only controls were identical before and after.

See `issue_replay.txt` for raw output.

## Existing regression coverage

`test/registered/unit/function_call/test_deepseekv4_detector.py` was introduced by the related fix and directly covers a prose preamble sharing a delta with a tool call. All six tests pass in the prepared interpreter. See `deepseekv4_tests.txt`.

## Hardware and scope

The assigned device is one AMD Instinct MI350X (`gfx950`). This bug is in deterministic CPU-side response parsing, so GPU execution would not add evidence and was not performed. The original DeepSeek V4 weights and four-H200 CUDA topology were unavailable. Consequently, this result does not claim full-model generation, DSPARK/speculative decoding, CUDA, or distributed reproduction. The tiny Llama serving fixture is not a valid substitute for DeepSeek V4 formatting or semantics.
