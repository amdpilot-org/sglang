# Caller-provided multimodal cache IDs

Upstream issue: https://github.com/sgl-project/sglang/issues/38651

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2670

## Baseline reproduction

At base commit `358c163250ad3b1f62939b01ce1314a0a31a0365`, Pydantic accepted an
unknown `cache_id` in Chat Completions media objects but silently discarded it.
For example, validating
`{"type":"video_url","video_url":{"url":"x","cache_id":"stable"}}`
produced `{"type":"video_url","video_url":{"url":"x", ...}}` without the
identity. Responses `input_image` validation likewise discarded `cache_id`.

## Implemented candidate

- Adds validated opaque `cache_id` fields for image, video, and audio Chat
  Completions parts and preserves them while converting Chat/Responses inputs.
- Adds native `mm_cache_ids` for image inputs and carries inline identities in
  `ImageData`, `VideoData`, and string-compatible `AudioData` values.
- Domain-separates opaque IDs before using them in the existing SHA-256 artifact
  key namespace. The artifact key continues to include modality, processor/model
  fingerprint, and preprocessing parameters.
- Reuses the existing `--trust-mm-content-hashes` opt-in. With trust disabled,
  `cache_id` does not bypass strict content reads or hashing. With trust enabled,
  a hot artifact hit occurs before source I/O. A miss reads and preprocesses the
  source normally and stores the artifact under the caller identity.
- Rejects empty IDs, IDs over 1024 UTF-8 bytes, misaligned native ID arrays, and
  ambiguous items that specify both `content_hash` and `cache_id`.

## Scope and limitations

The shared artifact-cache implementation is modality-neutral, but at this base
commit only Kimi-K3 image processing is connected to it. Therefore the no-I/O
hot-hit behavior is verified through the shared cache contract and Kimi-K3 image
wiring, not with a video/audio model. Video and audio API identities are parsed
and retained for processors adopting the shared artifact cache, but current
video/audio processors still perform their legacy preprocessing. No compatible
multimodal model weights were present, so no serving-path or GPU model execution
was claimed. This is a candidate implementation, not a claim that every model
architecture now has preprocessing-cache support.

No native code changed; no native rebuild is applicable.
