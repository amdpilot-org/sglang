# Independent review of PR 2620

Candidate reviewed: https://github.com/amdpilot-org/sglang/pull/2620 at exact commit `208bb69de35875426937fbf28ec209ed4f0c251c`.

Upstream issue: https://github.com/sgl-project/sglang/issues/23363

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2622

## Recommendation

Request changes. The candidate fixes both concrete counterexamples inherited from PR 2595: the complete-first/unclosed-second single-increment overflow is released, and singular K2-Thinking section markers work in `has_tool_call` and non-streaming parsing. Its focused parser suite passes (70 tests plus 7 subtests).

It does not fully resolve the original issue's streaming contract. With `SGLANG_KIMI_PARSER_SECTION_MAX=128`, an unclosed call delivered across several increments emits argument deltas before crossing the limit, then emits the same call payload as normal text on overflow. Consumers therefore receive an irrevocable partial tool call plus duplicated/mixed normal content rather than a clean safety-valve fallback. Separately, a valid JSON string containing the literal `<|tool_call_end|>` token is prematurely terminated by the streaming parser, producing truncated invalid JSON and leaking the suffix as normal text. Non-streaming parsing handles that same input correctly.

## Reproduction

The prepared interpreter and actual source module were used throughout:

```text
/tmp/amdpilot-repo-j-7744daa0edeb/venv/bin/python
/job/repo/python/sglang/srt/function_call/kimik2_detector.py
```

Commands:

```bash
git switch --detach 358c163250ad3b1f62939b01ce1314a0a31a0365
PYTHONPATH=/job/repo/python /tmp/amdpilot-repo-j-7744daa0edeb/venv/bin/python /job/review_harness.py

git switch --detach 208bb69de35875426937fbf28ec209ed4f0c251c
PYTHONPATH=/job/repo/python /tmp/amdpilot-repo-j-7744daa0edeb/venv/bin/python /job/review_harness.py
PYTHONPATH=/job/repo/python /tmp/amdpilot-repo-j-7744daa0edeb/venv/bin/python -m pytest -q test/registered/function_call/test_kimik2_detector.py
```

Raw outputs and the independent harness are retained outside the revision-switched checkout under `/job/review-evidence/` and `/job/review_harness.py`.

## Scope and limitations

This is deterministic parser logic and changes no native source, so no native rebuild was applicable. No GPU execution was needed or performed. The assigned gfx950 architecture, a real Kimi model, model weights, and an HTTP serving run were not used; therefore this review establishes parser behavior only and does not claim model-semantic or end-to-end serving validation.
