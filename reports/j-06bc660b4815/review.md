# Independent review of PR 1613

Candidate: https://github.com/amdpilot-org/sglang/pull/1613

Exact candidate commit: `d395e6f562c9780e2f0906e5f341163830949ba7`

Upstream issue: https://github.com/sgl-project/sglang/issues/35433

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1653

## Recommendation

Accept. The candidate fully resolves the original issue by choosing the explicitly
allowed rejection behavior: DeepSeek-V4 conversations reject system messages that
occur after any non-system message. It also closes the independently reported
system-only counterexample, which otherwise produces no assistant generation
boundary.

This is a source fix with regression hardening, not a test-only change. The new
`validate_system_message_order(context + messages)` call executes before prompt
rendering and tokenization. The common serving request wrapper converts its
`ValueError` into HTTP 400.

## Independent evidence

The prepared checkout was exactly the recorded base
`358c163250ad3b1f62939b01ce1314a0a31a0365`. With imports resolving to
`/job/repo/python/sglang/srt/entrypoints/openai/encoding_dsv4.py`, the base:

- accepted the issue's trailing-system tool-call conversation and rendered no
  `<｜Assistant｜><think>` suffix;
- accepted `[{'role': 'system', 'content': 'a'}]` as
  `'<｜begin▁of▁sentence｜>a'` without a generation boundary;
- accepted developer-then-system and context-user/new-system variants without a
  generation boundary.

At exact candidate commit `d395e6f...`, all four forms raise `ValueError` before
tokenization. Independent boundaries also established that:

- multiple leading system messages followed by a user remain accepted and end in
  `<｜Assistant｜><think>`;
- a leading system message in context followed by a new user remains accepted;
- user-only requests remain accepted with the correct generation boundary;
- a system-only context plus a new system message is rejected;
- unrelated custom `latest_reminder` behavior is unchanged.

The candidate's 11 focused DSV4 tests and the complete 138-test
`test_serving_chat.py` module passed. Raw logs, source diffs, issue/PR snapshots,
and the independent matrix are retained outside the checkout at
`/job/review-evidence-j-06bc660b4815/` so revision switching did not overwrite
them.

## Source, native, and environment qualification

No native, C++, HIP, FlyDSL, or build-system file changes between the recorded
base and candidate, so a native rebuild is not applicable. Python imports were
confirmed from the candidate checkout rather than an installed SGLang wheel.
Torch resolves from the prepared ROCm environment (`2.11.0+rocm7.2`, HIP 7.2).

The host exposes one AMD Instinct MI350X GPU, but no DeepSeek-V4-Flash-0731 weights
were available. No GPU model execution was performed, and the reported long-context
first-token-EOS symptom was not replayed. That architecture/model limitation does
not prevent qualification of this fix because the accepted contract is rejection
before tokenizer/model execution; the deterministic source and serving path were
exercised directly. The tiny Llama fixture was not substituted because it cannot
qualify DeepSeek-V4 prompt semantics.
