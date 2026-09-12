# Independent review of PR 3081 at `5a5fa419998d16bb1c1cba5a91023b1272e06444`

Upstream issue: https://github.com/sgl-project/sglang/issues/1932

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/3040

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/3110

## Recommendation

Request changes. The candidate is a functioning partial optimization for repeated,
identical Qwen image requests with trusted SHA-256 content hashes. It does not fully
implement the original request to save expensive media preprocessing and load it by
ID for another request: the rendered prompt is part of the cache key, there is no
explicit save/load API or caller-visible cache ID lifecycle, and entries are local
to one tokenizer worker and lost on restart. Simultaneous identical cold requests
also each preprocess because this new path uses `get`/`put`, not the cache's
single-flight interface.

## Evidence

- Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`.
- Candidate: `5a5fa419998d16bb1c1cba5a91023b1272e06444`.
- The GitHub PR reports base OID `bd45cd50ca900dd821f829ca9adfbf9aa3336bda`,
  which differs from the campaign's recorded failing-before base. This review used
  the required recorded base for comparison.
- On the recorded base, the candidate regression file fails all three tests because
  `QwenVLImageProcessor` has no `_request_preprocess_cache_key`.
- On the exact candidate, its regression plus preprocess-cache and hash tests pass:
  38 tests and 5 subtests.
- Imports resolved to `/job/repo/python/sglang/...`, confirming the source checkout
  under review was exercised.
- No C/C++/HIP/native source changed, and `repository-environment.json` declares no
  native build target, so no native rebuild applies.
- An independent adversarial check demonstrates that the same image content ID with
  a changed prompt produces a different key, and two simultaneous identical cold
  misses both enter preprocessing.
- `git diff --check` passes.

Raw command output, issue/PR JSON, the candidate patch, import paths, and environment
inventory were preserved outside revision switches at
`/job/review-evidence-j-2ce098b8d7d5/`.

## Architecture and environment limits

The assigned runtime exposes ROCm 7.2 and Torch `2.11.0+rocm7.2`; GPU inventory was
recorded, but no qualifying Qwen2-VL/Qwen2.5-VL weights were provided. The available
tiny Llama fixture cannot validate Qwen multimodal preprocessing, semantic accuracy,
or distributed/tokenizer-worker sharing. Therefore no GPU numerical or end-to-end
Qwen serving claim is made. The reviewed change is Python-side cache control flow and
does not modify native code.

