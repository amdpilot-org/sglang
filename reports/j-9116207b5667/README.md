# Independent review of PR 1026

Reviewed https://github.com/amdpilot-org/sglang/pull/1026 at exact commit
`6530140b906df8d7d24180aa99d6b43c737d79f8` against recorded base
`358c163250ad3b1f62939b01ce1314a0a31a0365` and the original issue contract.

Recommendation: **accept as a narrow partial parser fix**. The candidate does
not fully resolve the original issue, and its own prose correctly avoids that
claim.

The base retained clean model-added `arguments` and `args` envelopes. The exact
candidate normalized object-valued, JSON-string-valued, alternate-key, mixed
repeated, and repeatedly string-encoded envelopes when the declared tool schema
made the envelope unambiguous. Independent one-shot and streaming checks passed
with chunk sizes 1, 2, 3, 7, and 64. The focused candidate suite passed 11 tests
and 10 subtests; the broader function-call parser suite passed 243 tests and 4
subtests.

The following original-issue behaviors remain outside the repair:

- duplicated or corrupted inner command text remains corrupted after envelope
  removal because the intended command is absent from the payload;
- syntactically invalid inner JSON strings remain strings after all safely
  removable outer envelopes are peeled;
- schemas without usable declared top-level properties retain the envelope,
  because `arguments` or `args` may be a legitimate free-form key;
- model retry escalation and generation-time corruption were not reproduced.

The imported module path was
`/job/repo/python/sglang/srt/function_call/deepseekv4_detector.py`. The diff is
Python-only, so no native rebuild was applicable. The assigned host exposes a
gfx950 / AMD Instinct MI350X, but no GPU execution was needed for this parser
review. The named `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` weights were not
available; therefore model architecture, vision semantics, generation
frequency, and end-to-end retry behavior remain unverified. The unrelated tiny
Llama fixture was not used as substitute evidence.

Upstream issue: https://github.com/sgl-project/sglang/issues/38013

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1061
