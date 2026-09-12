# Independent review of PR 3235

Reviewed exact candidate commit `4b29cbe7f919d23dd82ff76f65257a85a72889a2`
against base `358c163250ad3b1f62939b01ce1314a0a31a0365` and the original issue.

## Verdict

Request changes. The candidate is a valid partial optimization for repeated,
identical Qwen image requests and fixes the previously reported concurrent
cold-request race with single-flight lookup. It does not implement the original
request to explicitly save preprocessing and later load it by a caller-selected
ID for another prompt.

Independent adversarial execution showed that the same content ID with a changed
rendered prompt produces a different key and performs preprocessing twice. A
fresh processor instance also performs preprocessing again, matching the source
comment that each tokenizer worker owns a separate process-local cache. Entries
do not survive restart. Video and audio explicitly bypass this Qwen cache path.

## Evidence

- `raw/base-regression.log`: the candidate regression run on the recorded base
  immediately failed; the later single-flight test could not progress because
  the base has no candidate cache entrypoint, so the run was terminated rather
  than misreported as a completed suite.
- `raw/candidate-regression.log`: 4 passed at the exact candidate.
- `raw/adversarial.log`: prompt change caused two preprocessing calls; a second
  processor instance also recomputed.
- `raw/independent-suite.log`: 70 passed and 30 subtests passed for the existing
  preprocessing-cache and OpenAI protocol suites.
- `raw/import-path.log`: imports resolved to the checked-out source under
  `/job/repo/python`, while AIter loaded from the prepared private runtime cache.
- `raw/gpu.log`: one AMD Instinct MI350X (`gfx950`) was visible.

No native source changed, so no native rebuild was applicable. No Qwen-VL model
weights were available. The supplied tiny Llama fixture cannot qualify Qwen-VL
preprocessing, semantic accuracy, or performance, so GPU execution was not used
as evidence for this review.

Upstream issue: https://github.com/sgl-project/sglang/issues/1932

Candidate PR: https://github.com/amdpilot-org/sglang/pull/3235

Mirror issue: https://github.com/amdpilot-org/sglang/issues/3238
