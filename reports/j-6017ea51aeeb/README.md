# Independent review of PR 2519

Candidate commit: `a8cd992b18403cd79c7b1defed0d6862f91f813c`

The candidate is a correct narrow fix for a fragmented
`<|tool_call_end|>` marker: the recorded base emits the marker prefix as JSON
argument text, while the candidate holds it until it completes or diverges.
Its focused suite (63 tests plus 7 subtests) and legacy KimiK2 selection (6
tests) pass from the repository source path.

It does not fully resolve the original open issue. Independent cases still
show that:

- one-character fragmentation of all markers emits marker text as content and
  produces no calls;
- an unclosed 600,000-character section remains buffered in full, with no
  requested bounded safety valve or normal-text flush;
- singular K2-Thinking section markers are left in `normal_text`.

The current base already parses the tested long nested JSON in non-streaming
mode and supports two long streaming calls when structural begin markers are
delivered intact. Thus the candidate is a partial original-issue fix, not a
full rewrite or full resolution.

No native files changed, so no native rebuild applies. The imported candidate
module was `/job/repo/python/sglang/srt/function_call/kimik2_detector.py`, not a
wheel copy. Testing was deterministic CPU parser testing. The available host
reports ROCm 7.2 and one AMD Instinct MI350X (gfx950), not the issue's
8x H100/H200/B200 systems; no Kimi weights were available, so model semantics,
HTTP serving, client behavior, and distributed execution remain unverified.
