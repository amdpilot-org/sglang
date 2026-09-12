# Independent review of PR 528 at c3469e06635506f3c36836b342ef488e964b8201

Upstream issue: https://github.com/sgl-project/sglang/issues/39103

Mirror issue: https://github.com/amdpilot-org/sglang/issues/644

Candidate: https://github.com/amdpilot-org/sglang/pull/528

## Verdict

Recommendation: accept. The candidate fully resolves the original issue within the exercised OpenAI-compatible conversion contract. It is a source fix with regression coverage, not test-only hardening. No remaining counterexample was found.

## Evidence

The prepared branch was initially at base `358c163250ad3b1f62939b01ce1314a0a31a0365`. I overlaid only the candidate's four test files while retaining base implementation source and ran the six new `include_reasoning` cases. All six failed: chat leaked `reasoning_content`; Responses emitted reasoning items/events; legacy completions retained `<think>private</think>` in both response modes. This independently reproduces the original defect in the actual response builders and stream converters.

I then restored the tree and detached at the exact candidate commit. The full four-file endpoint suite passed with `204 passed, 72 subtests passed`. These tests cover default/opt-in behavior, false/true cross-request isolation, non-streaming and streaming chat/Responses/completions, and preservation of Responses tool calls.

Independent temporary adversarial tests (not committed to the review branch) exercised legacy completion streaming with reasoning delimiters divided across multiple chunks. Both cumulative and incremental engine-output modes produced only `answer`, and `echo=True` produced `PROMPTanswer` without leaking reasoning. All three passed.

Import provenance at the candidate resolved `sglang` and every changed serving module under `/job/repo/python/sglang/`. Torch resolved from the prepared environment as `2.11.0+rocm7.2`, HIP `7.2.26015`. One AMD Instinct MI355X was visible. A float64 GPU product exactly matched an independently constructed CPU reference (`max_abs_error=0.0`).

The diff contains only Python, tests, and the candidate's report artifacts; it contains no C/C++/CUDA/HIP source. `repository-environment.json` also records `native: null`, so a native rebuild was neither required nor applicable.

## Limitations

No model weights were included in the prepared environment, so I could not launch a live reasoning-model server and repeat the issue's curl requests end to end. The endpoint tests drive the actual request schemas, reasoning/tool parsers, response builders, and streaming converters with synthetic engine output; they are stronger than plain-text generation smoke tests but do not validate model-specific output behavior.

Raw command output was retained outside revision switches under `/job/review-evidence/`. The review checkout was returned to `amdpilot/j-48638a0e63cd` before this report was committed.
