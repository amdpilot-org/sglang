# Independent review of PR 1169

Upstream issue: https://github.com/sgl-project/sglang/issues/36675

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1208

Candidate: https://github.com/amdpilot-org/sglang/pull/1169 at exact commit `22a689112e3e8fab6b3b1206ac6e5dbed54c871b`

## Recommendation

Accept. The candidate fully resolves the source-level defect described by the original issue. At recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`, `_get_reasoning_from_request()` returns false whenever no reasoning parser is configured, while `xgrammar_reasoning` also requires the parser to be absent. The candidate separates template reasoning state from parser-owned output state and makes the intended parser-less path reachable without regressing parser ownership.

This is a functional fix with a failing-before/passing-after regression, not test-only hardening. No remaining counterexample was found within the original contract.

## Independent evidence

- The prepared checkout started clean on the recorded base; there was no difference between the image-prepared checkout and required failing-before commit.
- With the candidate regression test temporarily applied to the base, the two enabled parser-less cases failed because `thinking_mode` was observed as false. The two disabled cases passed.
- At the exact candidate commit, the regression and existing parser-owned boundary passed: 2 tests plus 4 subtests.
- Independent adversarial coverage passed for an always-on parser-less template, Mistral effort off/on behavior, absent/incomplete template reasoning metadata, and a configured parser that must retain ownership of the reasoning prefix.
- The complete `test_serving_chat.py` file passed: 136 tests plus 72 subtests.
- An actual `FunctionCallParser(..., "qwen3_coder")` invocation showed that `thinking_mode=False` produces a tool-call-only `tags_with_separator`, while `thinking_mode=True` produces a `sequence` whose first element permits arbitrary text ending at `</think>` before the tool-call grammar.
- `git diff --check` passed.

## Source, native, and architecture notes

The interpreter loaded `sglang` and `serving_chat.py` from `/job/repo/python/sglang`, so candidate execution used the checked-out source rather than an installed SGLang wheel. Torch remained `/opt/venv/lib/python3.12/site-packages/torch`, version `2.11.0+rocm7.2` with HIP `7.2.26015`.

The candidate changes only Python serving logic, a Python unit test, and reports. It changes no native source, so no native rebuild applies. One assigned AMD Instinct MI350X was visible with capability `(9, 5)` (`gfx950`), but GPU execution is not relevant to this pre-generation Python control-flow and structural-tag construction fix and was not used as proof.

Qwen3.5 weights were unavailable, and the registered Qwen3.5 model test requires eight GPUs. Therefore no full-model HTTP, semantic-accuracy, or distributed reproduction was performed. The qualified tiny Llama fixture cannot validate Qwen3.5 chat-template/reasoning behavior and was intentionally not substituted. These are architecture-level validation limitations, not source-level counterexamples to the fix.
