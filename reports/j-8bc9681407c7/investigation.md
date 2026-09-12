# Independent review of PR 1773

Reviewed exact candidate commit `82c3e4430316be3a3928e3bd2a7bb445a14f4e87` against recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`.

Recommendation: **accept**. The candidate fully resolves the original issue within its deterministic Python-state scope. No remaining counterexample was found.

## Independent evidence

The prepared checkout was clean and exactly at the recorded base. With `PYTHONPATH=/job/repo/python`, Python reported the loaded implementation as `/job/repo/python/sglang/srt/entrypoints/openai/streaming_asr.py`, excluding an installed-wheel false positive.

At the base, the original reproduction emitted `erpillar` and accumulated `the cat erpillar`. The punctuation case returned `,` but accumulated `hello world ,`. Both match the issue report.

At the exact candidate, the supplied focused suite passed (`37 passed, 19 warnings, 11 subtests passed`). An independent script additionally checked update and finalization with comma, quote, Unicode ellipsis, em dash, `?!`, fullwidth closing parenthesis, Arabic semicolon, ideographic full stop, guillemet, and multi-character CJK punctuation. It also checked ordinary word append, a final complete-word extension, and `process_asr_chunk(..., is_last=True)`. Every assertion passed.

The candidate is a source fix with regression hardening, not a test-only change: `streaming_asr.py` adds boundary-aware append/rollback classification and punctuation-aware accumulation. No native code or build file changed, so a native rebuild was not applicable.

## Scope and limitations

This review did not use Qwen3-ASR weights, run a live HTTP/realtime server, or claim semantic model accuracy. The deterministic fake tokenizer manager exercises the actual shared `process_asr_chunk` helper and its finalization path, but not network transport. GPU execution was neither needed nor performed. The prepared environment identifies Torch `2.11.0+rocm7.2` and HIP `7.2`; no architecture-specific conclusion is drawn.

The class documents an existing whitespace-free CJK rollback limitation, which is outside this issue's word-growth and punctuation-attachment contract. Punctuation rollback also retains the already-emitted monotonic prompt history; this was explicitly tested by the candidate and does not reproduce the reported append-growth defect.

Raw command transcripts were preserved outside the revision-switched checkout at `/job/review-evidence-j-8bc9681407c7/` during review. The checkout was returned to `amdpilot/j-8bc9681407c7` before this report was added.
