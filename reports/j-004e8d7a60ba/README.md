# Independent review of MiniMax-M3 top-level `oneOf` coverage

Candidate: https://github.com/amdpilot-org/sglang/pull/2139 at
`d6440ada6c4720260a5833ece67cafc3ed5832db`

Upstream issue: https://github.com/sgl-project/sglang/issues/32286

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2181

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/2103

## Verdict

Recommendation: **accept**. The candidate is useful test-only hardening, not a
new source fix. The recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`
already uses recursive `get_schema_properties()` lookup in
`MinimaxM3Detector._get_child_schema()`. The exact candidate regression passes
against that implementation and independently reproduces the reported numeric
string and raw-array-tag failures when the parser's lookup is replaced with the
former top-level-only behavior.

This fully resolves the original issue's specific schema: a root containing
only `oneOf`, with distinct `number`, `stringList`, and `numberList` properties
in its branches. It does not establish unrestricted support for every legal
combination of JSON Schema composition keywords.

## Evidence

- Prepared checkout before review: clean branch
  `amdpilot/j-004e8d7a60ba` at the exact recorded base.
- Exact candidate checkout: detached
  `d6440ada6c4720260a5833ece67cafc3ed5832db`; diff contains only tests and
  reports, with no Python implementation or native changes.
- Source imports resolved to `/job/repo/python/sglang/__init__.py`,
  `/job/repo/python/sglang/srt/function_call/minimax_m3.py`, and
  `/job/repo/python/sglang/srt/function_call/utils.py`.
- Base focused suite: 24 passed, 14 subtests passed.
- Candidate focused suite: 26 passed, 16 subtests passed.
- Independent legacy substitution: failed with `number == "42"` and both
  arrays containing raw `]<]minimax[>[<item>` tags, matching the report.
- Independent character-by-character streaming and non-stream parsing passed
  for `-4.25` and `[-1.5, 2e2]` under a root-only `oneOf`.
- Independent broader boundary: a legal schema with both root `properties` and
  root `oneOf` leaves the branch-specific number as string `"3.5"`. The helper
  returns root properties before traversing composition branches. This is a
  remaining counterexample to a generalized composition-support claim, but is
  not the original issue's schema.

Raw command output and the independent scripts are retained outside the
checkout at `/job/review-evidence-j-004e8d7a60ba/` so revision switches did not
overwrite them.

## Architecture and environment limitations

The assigned hardware is one AMD Instinct MI350X (`gfx950`). Parser tests are
deterministic CPU-side tests and did not execute GPU kernels. MiniMax-M3 weights
were unavailable, and the issue's command requires four GPUs, so no full-model,
HTTP, semantic-accuracy, or distributed verifier run is claimed. The tiny Llama
fixture cannot qualify MiniMax-M3 output and was not used. No native source
changed; `repository-environment.json` reports `native: null`, so no native
rebuild was applicable.
