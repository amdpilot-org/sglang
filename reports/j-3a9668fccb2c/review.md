# Independent review of PR 1583

Candidate: https://github.com/amdpilot-org/sglang/pull/1583

Exact commit: `bd84efaef10c01b4354e5ceed7b69a6c247d9ec5`

Upstream issue: https://github.com/sgl-project/sglang/issues/35564

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1618

## Recommendation

Request changes. The candidate is a substantial partial fix and passes both its focused regression and the repository's function-call suite, but it does not fully satisfy the original parity contract.

## Reproduction and verification

The recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365` was the prepared checkout, with no difference from the requested base. Running the issue's exact reproduction there exited 1 and reproduced GLM/GLM45/GLM47 numeric type corruption, the missing MiniMax call, and Step3's extra call. The reproduction's dictionary aggregation masks the reported Cohere/Gemma index defect and collapses Mistral calls that share an index, so its `OK` labels for those cases are not independent proof.

At the exact candidate commit, imports resolved to `/job/repo/python/sglang/...`, including `glm4_moe_detector.py`, `glm47_moe_detector.py`, and `mistral_detector.py`. The issue reproduction exited 0. The candidate regression passed with 10 tests and 1,575 subtests, and the full function-call suite passed with 550 tests and 1,596 subtests. Independent GLM cases covering undeclared `null`, booleans, integers, floats, escaped quoted strings, arrays, and objects, plus declared string/number/boolean/null properties, matched one-shot parsing for whole, character, and two-way-split delivery.

## Remaining counterexample

Mistral streaming still assumes the literal separator `", "`, although JSON permits insignificant whitespace. For this valid three-call array with no space after commas:

```text
[TOOL_CALLS] [{"name":"get_weather","arguments":{}},{"name":"get_weather","arguments":{"n":2}},{"name":"f","arguments":{}}]
```

`detect_and_parse` returns calls 0, 1, and 2. Supplying the entire generation to `parse_streaming_increment`, followed by the serving-style two empty flushes, returns only call 0 and emits the remaining JSON objects as `normal_text`. Character-at-a-time and every two-way split fail the same independent check. Thus the candidate fixes the exact spaced fixture and the previously reviewed three-call case, but not the general original contract that streamed deltas equal one-shot parsing on the same accepted text.

## Environment and scope

The prepared environment uses Python from `/tmp/amdpilot-repo-j-3a9668fccb2c/venv/bin/python`, PyTorch 2.11.0+rocm7.2, and a gfx950-class ROCm environment rather than the reporter's CUDA/RTX 4090 setup. No GPU was used: this is deterministic Python string parsing. No native source changed, so no native rebuild was applicable. No model weights, tokenizer, HTTP serving process, semantic model behavior, multi-GPU, or distributed workload was exercised.
