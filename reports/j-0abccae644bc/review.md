# Independent review of amdpilot-org/sglang PR 1375

Candidate: `2b195c8caafa620378888215ed62d32d2babbd2a`

Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

Upstream issue: https://github.com/sgl-project/sglang/issues/35562

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1409

## Finding

Request changes. The candidate fixes the two reported examples and the common
case where one or more leading JSON objects decode to dictionaries but none is
a registered tool call. It also preserves valid unmarked tool-call extraction.

It does not fully implement the issue's stated fallback contract for arbitrary
false positives from the unmarked leading-`{` heuristic. The detector decodes
every semicolon-separated JSON value and calls `parse_base_json` before the new
`not calls` fallback. Consequently, ordinary content such as
`{"a":1};42` and `{"a":1};[]` raises `AttributeError` on the second JSON
value instead of being returned unchanged with no calls. Both failures occur
on the base and remain at the exact candidate commit. The candidate regression
does not cover non-object JSON values in the speculative unmarked path.

This is a partial original-issue fix, not test-only hardening: the source change
does correct the reported deletion. A narrow type-safe speculative parse (or
equivalent preservation before `parse_base_json`) and a failing regression for
non-object decoded values are still needed for the full contract.

## Evidence

The prepared checkout was exactly the recorded base. The candidate's parent and
merge base were also exactly that base. Imports resolved to the checked-out
source:

```text
sglang /job/repo/python/sglang/__init__.py
detector /job/repo/python/sglang/srt/function_call/llama32_detector.py
```

On the base, the public `FunctionCallParser.ToolCallParserEnum["llama3"]`
reproduction returned `"is a dict"` for `{"a": 1} is a dict` and an empty
string for `{}`. At the candidate, both inputs were returned byte-for-byte with
no calls. Independent passing cases included plain JSON, an unknown tool, two
non-tool objects, Python-dict syntax, a malformed text tail, and a genuine
registered tool call. The two non-object-tail cases above raised on both
revisions.

Candidate test results:

- `test/registered/unit/function_call/test_llama32_detector.py`: 18 passed and
  4 subtests passed.
- `test/registered/unit/function_call/test_function_call_parser.py -k
  Llama32Detector`: 7 passed, 236 deselected.
- Python compilation of the changed source succeeded.
- `git diff --check` found trailing whitespace only in the candidate's committed
  report artifacts, not in the parser or test source.

Complete revision-specific raw output is preserved outside the checkout at
`/job/review-evidence-j-0abccae644bc/`.

## Environment and architecture

The prepared Python was
`/tmp/amdpilot-repo-j-0abccae644bc/venv/bin/python`, with checkout source forced
through `PYTHONPATH=/job/repo/python`. The host exposes one gfx950 AMD Instinct
MI355X. No GPU execution, HTTP server, model weights, or distributed workload
was used: the defect and counterexample are deterministic CPU parser behavior.
No native files changed, so no native rebuild was applicable. This review does
not claim model semantic, serving transport, or multi-node validation.
