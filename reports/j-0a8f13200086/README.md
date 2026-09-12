# Independent review of PR 3310

Candidate: https://github.com/amdpilot-org/sglang/pull/3310

Exact commit: `2a64f7dfc428bb7b06ccf5187c5d6b7c81c90fa2`

Upstream issue: https://github.com/sgl-project/sglang/issues/1932

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/3313

## Verdict

Recommendation: **request changes**. The candidate is a useful but partial
optimization for repeated, identical Qwen image requests. It does not fully
implement the original request to explicitly save preprocessing and load it by
a caller-selected ID on a later request. The candidate's prose correctly
acknowledges that limitation, so this is not an unverified full-fix claim.

The exact candidate passed its four regression tests and the focused cache,
parser, protocol, and identity suite (115 tests plus 30 subtests). Independent
adversarial execution nevertheless confirmed:

- changing only the rendered prompt for the same content ID produces a new key
  and performs preprocessing again;
- a new processor instance has an empty cache and recomputes;
- neither the Chat Completions nor Responses request model exposes an explicit
  preprocessing save/load operation or cache ID;
- Qwen video and audio requests return no request-cache key.

The candidate also claims `git diff --check` passed, but running it against the
recorded base and exact candidate found trailing whitespace in two committed
raw pytest logs. This does not affect runtime behavior, but the recorded claim
is inaccurate and should be corrected.

## Reproduction

The prepared base was `358c163250ad3b1f62939b01ce1314a0a31a0365`.
On that revision, runtime inspection showed that
`QwenVLImageProcessor.process_mm_data_async` called `load_mm_data` directly,
had no `_request_preprocess_cache_key`, and did not reference
`mm_preprocess_cache`. This reproduces the original absence of Qwen request
preprocessing reuse.

After temporarily checking out the exact candidate, imports resolved to:

```text
sglang /job/repo/python/sglang/__init__.py
qwen_vl /job/repo/python/sglang/srt/multimodal/processors/qwen_vl.py
```

Commands and measured results are recorded in `result.json` and
`raw/review.log`. The checkout was returned to
`amdpilot/j-0a8f13200086` before this report was committed.

## Architecture and environment limits

No native source changed, so no native rebuild was applicable. Torch remained
the prepared ROCm build (`2.11.0+rocm7.2`, HIP 7.2). No qualifying Qwen-VL
weights were available, so no Qwen GPU semantic, numerical, or performance
claim was made. The tiny Llama fixture is not capable of validating Qwen media
preprocessing semantics, multi-worker ownership, or persistence. The tested
unit paths execute the real checked-out Python implementation with mocked Qwen
preprocessing results; they do not prove end-to-end model equivalence.
