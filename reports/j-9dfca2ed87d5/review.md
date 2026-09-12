# Independent review of candidate PR 1390

Reviewed exact candidate commit `a99886a78d232e0d0bf3c7034f66061c26002874` against recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365` and the original issue contract.

Recommendation: **accept**. The candidate fully resolves the original direct-detector text-loss cases.

## Evidence

- The prepared branch initially and finally pointed at the recorded base commit.
- On the base, an independent matrix reproduced text loss for both the bare marker and `visible-prefix + marker` for all 13 named parsers: 26 failures, exit 1.
- On the exact candidate, the same matrix passed all cases. It also checked marker-like ordinary suffix text and unknown-function/unparseable blocks; when no call was produced, the complete input was retained.
- Candidate-focused tests passed: 265 tests and 17 subtests. These include the 13-parser regression, a complete valid Hermes call (still parsed as a call), plain text, prefixed truncation, and Mistral invalid JSON.
- Checkout imports were confirmed as `/job/repo/python/sglang/__init__.py` and `/job/repo/python/sglang/srt/function_call/base_format_detector.py`, not an installed SGLang copy.
- The candidate changes only Python parser code, tests, and its prior report artifacts. It has no native source change, so no native rebuild was applicable.

Raw logs, the independent script, issue/PR metadata, import paths, and the exact candidate diff are retained outside the checkout in `/job/review-evidence-j-9dfca2ed87d5/`.

## Scope and limitations

This is deterministic CPU text parsing. No GPU, model weights, HTTP server, semantic-accuracy test, or distributed workload was used or claimed. The available environment is ROCm 7.2 on one gfx950 GPU, unlike the CUDA RTX 4090 source environment, but the reviewed path does not dispatch to either architecture.

The open related upstream PR https://github.com/sgl-project/sglang/pull/35636 changes the `FunctionCallParser` wrapper. The reviewed candidate instead fixes the detector-level behavior used by the issue reproducer, so its evidence is not dependent on that wrapper-only change.

No remaining counterexample was found within the original contract.
