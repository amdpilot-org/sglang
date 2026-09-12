# Independent review of amdpilot-org/sglang PR 808

Upstream issue: https://github.com/sgl-project/sglang/issues/38821

Mirror issue: https://github.com/amdpilot-org/sglang/issues/853

Candidate: https://github.com/amdpilot-org/sglang/pull/808 at `436f0c1f6d0b64fa2947441bdd0add9611b4778b`

Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

## Recommendation

Request changes. The candidate is a partial processor-registration fix, not a
verified resolution of the original semantic vision failure.

The recorded base reproduces a related prerequisite failure for the public
`zai-org/GLM-5.3-Flash` checkpoint: SGLang resolves a tokenizer-only
`TokenizersBackend`, and accessing `image_processor` raises `AttributeError`.
At the candidate revision, the same real loading path resolves
`Glm5NextProcessor` and `Glm5NextImageProcessor`, and a full processor call on
a synthetic JPEG produces a finite `(80, 1176)` tensor and a `1 x 8 x 10`
grid. The candidate's six focused tests pass.

However, the candidate does not include the serving-path normalization needed
when GLM-5.3 processor-expanded input IDs are supplied with the original image.
Its processor expands one image placeholder into a repeated image-token span;
`Glm4vImageProcessor.process_mm_data_async` passes that span unchanged to
`load_mm_data`, where it can be interpreted as multiple image placeholders.
An independent regression using one image and four expanded tokens fails on
the exact candidate: the observed prompt is `[1, 10, 99, 99, 99, 99, 11, 2]`
instead of the required collapsed `[1, 10, 99, 11, 2]`. The directly related
upstream PR 36833 contains this omitted normalization and integration coverage.

More importantly, the original issue reports semantically wrong output from a
deployment that already accepted and processed an image. Registering a missing
processor in this prepared base does not establish why that deployment saw a
bird, nor does a solid-color preprocessing shape test prove that a Statue of
Liberty JPEG is interpreted correctly. The exact JPEG (SHA-256
`ff13fd6f991b37253d3745dc6b9ef8e7a92f17cee7c4c8bc84735000a668fcd7`), private
`GLM-5.3-Flash-sglang0907-128k-0907t3` weights, and original deployment were
unavailable.

## Evidence

- Base, real source import: `sglang.srt.utils.hf_transformers.get_processor`
  returned `TokenizersBackend`; `image_processor` access failed.
- Candidate source import paths resolved to `/job/repo/python/sglang` and
  `/job/repo/python/sglang/srt/configs/glm5_next_processing.py`, confirming the
  checkout source rather than an installed SGLang wheel was tested.
- Candidate regression: `6 passed` in
  `test/registered/unit/multimodal/test_glm5_next_processor.py`.
- Candidate full processor probe: returned `Glm5NextProcessor`,
  `Glm5NextImageProcessor`, `pixel_values (80, 1176)`, grid `[[1, 8, 10]]`, and
  20 image tokens.
- Independent serving-path regression: failed because `load_mm_data(prompt=)`
  received four consecutive image tokens for one image rather than one
  placeholder.
- No C/C++/CUDA/HIP/native files differ between base and candidate, so no native
  rebuild was applicable.

## Environment limitations

The assigned accelerator is one AMD Instinct MI355X (`gfx950`) with ROCm 7.2
and Torch `2.11.0+rocm7.2`, not eight NVIDIA H20 GPUs. GPU inventory was
confirmed, but no GLM model weights were available and no issue-path model
inference ran on the GPU. The tiny Llama fixture is text-only and cannot verify
GLM-5.3 vision architecture or landmark semantics, so it was not substituted.
No claim is made about CUDA/H20, multi-node behavior, the private model variant,
or semantic accuracy.
