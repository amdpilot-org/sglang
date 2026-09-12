# Independent review of amdpilot-org/sglang PR 1407

Candidate reviewed: `855f19fa2122df8f27ca88aed2466af019cd8a7d`

Upstream issue: https://github.com/sgl-project/sglang/issues/35564

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1444

## Recommendation

Request changes. The candidate is a meaningful partial fix, but it does not fully satisfy the original contract that streamed deltas reconstruct the same calls as `detect_and_parse` for the same valid parser input.

The recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365` reproduced the reported GLM numeric typing, Minimax missing-call, and Step3 extra-call failures. The exact candidate passed all eight supplied literal examples and its targeted regression (`8 passed`, `876 subtests`). It also passed the complete existing function-call unit directory (`548 passed`, `897 subtests`). Imports were confirmed from `/job/repo/python/sglang`, so the checked-out candidate source, not an installed SGLang copy, was exercised.

Independent adversarial testing covered whole-input, character-at-a-time, and every two-chunk split for 24 nearby cases (1,729 comparisons before stopping each failing case at its first failing chunk mode). Seven cases failed:

- `glm`, `glm45`, and `glm47`: an undeclared `<arg_value>null</arg_value>` is `null`/`None` in one-shot parsing but the streamed result is the string `"null"`.
- `glm`, `glm45`, and `glm47`: an undeclared quoted literal `<arg_value>"abc"</arg_value>` is `"abc"` in one-shot parsing but the streamed result includes the quote characters as data (`'"abc"'`).
- `mistral`: a valid three-call `[TOOL_CALLS]` JSON array with `{}`, `{"n": 2}`, and `{}` arguments produces three calls one-shot, but a whole-input streaming increment produces only two calls and drops the second call's arguments and the third call entirely.

These failures are adjacent to the reported empty/simple-argument and repeated-call bug class, and directly violate the issue's stated equality contract. They are not model, tokenizer, GPU, or serving-smoke substitutions.

## Environment and scope

- Python: `/tmp/amdpilot-repo-j-d83907ed7d49/venv/bin/python`
- Source import: `/job/repo/python/sglang/__init__.py`
- Parser import: `/job/repo/python/sglang/srt/function_call/base_format_detector.py`
- Torch: `2.11.0+rocm7.2`; prepared host architecture is ROCm/gfx950.
- GPU execution: none. This bug is deterministic string/parser logic and needs no kernel execution. No model-serving, tokenizer-vocabulary, semantic-accuracy, full-model, or distributed claim is made.
- Native rebuild: not applicable. The candidate changes Python parser code and reports/tests only; no native source or extension changed.

Raw working evidence was preserved outside revision switching at `/job/review-evidence-j-d83907ed7d49/`.
