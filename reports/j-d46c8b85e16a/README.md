# Independent review of PR 3322

Candidate: https://github.com/amdpilot-org/sglang/pull/3322 at `a6d1911d49dd87c3052eed49a000c2c6817b8154`

Upstream issue: https://github.com/sgl-project/sglang/issues/38651

Mirror issue: https://github.com/amdpilot-org/sglang/issues/3345

## Recommendation

Request changes. The candidate is a useful partial fix, but it does not fully resolve the original feature contract.

The prepared base rejects `mm_cache_ids` entirely. The candidate accepts and normalizes caller IDs, fixes the previously reported native video-only alignment error, validates Chat image/video/audio IDs, and implements a genuine pre-I/O cache hit in the shared artifact cache. Its focused regression suite passes.

However, source inventory shows that only `KimiK3ImageProcessor` calls `prepare_media_artifacts`, and it is the only call passing `cache_ids`. Kimi K3 explicitly rejects video and audio. Thus Chat video/audio IDs survive parsing but cannot produce the promised repeated-request no-I/O hit. Other image processors also do not use this caller-ID artifact path. The Responses conversion retains an input-image ID, but the candidate contains no serving proof that `previous_response_id` avoids historical media loading, decoding, preprocessing, and feature hashing.

## Evidence

On base `358c163250ad3b1f62939b01ce1314a0a31a0365`, constructing `GenerateReqInput(..., video_data=[...], mm_cache_ids=[...])` fails with `TypeError: GenerateReqInput.__init__() got an unexpected keyword argument 'mm_cache_ids'`.

At the exact candidate, 94 focused tests plus 27 subtests pass, including its native video alignment regression and shared-cache synthetic no-I/O test. The OpenAI serving compatibility suites pass (170 tests plus 70 subtests), but they do not exercise a real multimodal model or prove historical-media no-I/O behavior.

Raw command output is retained under `evidence/`. The external revision-switch evidence was preserved at `/job/review-evidence/j-d46c8b85e16a/` during review.

## Environment and architecture limitations

The prepared interpreter uses Torch `2.11.0+rocm7.2`, HIP `7.2.26015`, and exposes one AMD Instinct MI350X. No compatible multimodal model weights were supplied. The deterministic tiny Llama fixture is transport/engine-only and cannot qualify image, video, audio, or Responses multimodal caching, so no GPU serving claim is made. No C++/HIP/native source changed in the candidate; native rebuild was not applicable.
