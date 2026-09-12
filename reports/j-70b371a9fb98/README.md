# Independent review of PR 2708

Upstream issue: https://github.com/sgl-project/sglang/issues/38651

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/2670

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2723

Candidate: https://github.com/amdpilot-org/sglang/pull/2708 at
`dcd46ee1a2e600bfd3cca89ff2e37f55a569d124`.

## Recommendation

`request_changes`. The candidate is a useful partial implementation, but it
does not fully resolve the original feature request. It validates and retains
`cache_id` for Chat image/video/audio parts, adds native image-oriented
`mm_cache_ids`, and demonstrates a real early artifact-cache hit for the Kimi
K3 image processor. The exact focused regression suite passes.

The original contract requires reusable identities and no-I/O hot hits for
images, videos, and audio across native, Chat Completions, and Responses APIs.
At this commit only `python/sglang/srt/multimodal/processors/kimi_k3.py` passes
`cache_ids` into `prepare_media_artifacts`. No video or audio processor uses the
artifact cache path. Accepting and retaining those fields therefore does not
make them operational. A native video-only request with one `mm_cache_ids`
entry is also rejected because normalization aligns the list against
`image_data` and reports `mm_cache_ids has 1 entries for 0 images`.

Responses support is likewise limited to normalizing `input_image.cache_id`.
There is no independently demonstrated Responses `previous_response_id`
multimodal run proving that historical media avoids I/O, decoding,
preprocessing, and feature hashing. The candidate itself reports that this was
not run.

## Evidence

The recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365` was the prepared
checkout, with no difference from the image-prepared revision. On that base,
Pydantic discarded `cache_id` from both a Chat video part and a Responses
`input_image`; importing `parse_cache_id` failed. This reproduces the missing
API behavior.

At the exact candidate commit:

```text
python -m compileall -q python/sglang/srt
PASS

python -m pytest -q \
  test/registered/unit/entrypoints/openai/test_protocol.py \
  test/registered/unit/parser/test_jinja_template_utils.py \
  test/registered/unit/managers/test_mm_hashes.py \
  test/registered/unit/multimodal/test_media_artifact_processor.py
91 passed, 27 subtests passed
```

Independent adversarial checks produced:

```text
schema_preservation: PASS
native_video_cache_id: FAIL 'mm_cache_ids has 1 entries for 0 images'
processor_cache_id_consumers: ['kimi_k3.py']
video_audio_hot_hit_contract: FAIL no video/audio processor consumes cache_ids
```

The source imports resolved to `/job/repo/python/sglang`, including the
candidate `identity.py`. Torch was the pinned `2.11.0+rocm7.2` build. The
environment exposes one AMD Instinct MI350X (`gfx950`, 270566162432 bytes VRAM).
No GPU model execution was claimed: compatible multimodal weights were not
provided, and the deterministic tiny Llama fixture cannot validate a
multimodal architecture or this cache contract.

No C++, HIP, FlyDSL, or other native source changed in the candidate, so a
native rebuild was not applicable. Importing the environment's existing AITER
resolved its core from the private prepared cache, not from a candidate native
build.

Raw command output, issue/PR JSON, the candidate diff, JUnit XML, source import
paths, and GPU inventory are preserved outside revision switches at
`/job/review-evidence-j-70b371a9fb98/`.

## Remaining work

- Wire caller identities through actual video and audio preprocessing/cache
  implementations and prove second-turn no-I/O hits.
- Define and implement native modality alignment instead of treating
  `mm_cache_ids` as image-only while exposing video/audio APIs.
- Exercise a real multimodal model through Chat and Responses, including
  `previous_response_id`, cache eviction fallback, and independent I/O/decode/
  preprocess/hash instrumentation.
- Confirm coverage beyond Kimi K3 images or explicitly narrow the public API
  and feature claim.
