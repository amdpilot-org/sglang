# Independent review of PR 2581

Reviewed https://github.com/amdpilot-org/sglang/pull/2581 at exact commit `6791b29b61cb0054a6be3387528f08f31d1872f8` against https://github.com/sgl-project/sglang/issues/23363 and recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`.

Recommendation: **request changes**. The candidate is a real partial fix, not test-only hardening: its focused suite passes and independent tests confirm repaired one-character streaming fragmentation, streaming singular-marker stripping, simple overflow release, and explicit reset. It does not fully resolve the original contract.

## Remaining failures

1. The safety check runs only once before the parsing loop and asks whether *any* end marker exists in the buffer. A complete first call therefore masks an oversized unclosed second call in the same increment. With a configured limit of 128, the independent case retained 269 characters. Calling the production `finish()` method returned no text and did not clear the buffer.
2. Singular K2-Thinking section markers were added only to stripping and streaming partial-marker recognition. `has_tool_call()` and `detect_and_parse()` still require the plural section-begin marker, so the complete non-streaming singular format yields no call and leaks the wire markers as normal text.

Direct detector reuse without `reset()` still advances the next request from tool index 1, but inspected serving paths instantiate parsers per response choice. The new explicit `reset()` correctly restores index 0 when a direct caller uses it, so this is recorded as a limitation rather than an additional rejection counterexample.

## Environment

- Source import confirmed as `/job/repo/python/sglang/srt/function_call/kimik2_detector.py` at the exact candidate checkout.
- Python 3.12.3, Torch 2.11.0+rocm7.2, HIP 7.2.26015.
- One GPU was visible, but GPU execution is not relevant to this Python-only deterministic parser review and was not performed.
- No native files changed; `repository-environment.json` reports no native artifact or wheel, so no rebuild was applicable.
- No Kimi weights were available. Model semantics, actual token generation, HTTP behavior, and distributed execution remain unverified.

Raw command output is retained in `base-harness.txt`, `candidate-harness.txt`, and `candidate-tests.txt`.
