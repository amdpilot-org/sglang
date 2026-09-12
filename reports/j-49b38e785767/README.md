# Correction review: caller-provided multimodal cache IDs

Upstream issue: https://github.com/sgl-project/sglang/issues/38651

Mirror issue: https://github.com/amdpilot-org/sglang/issues/3306

Candidate parent: https://github.com/amdpilot-org/sglang/pull/2708 at
`dcd46ee1a2e600bfd3cca89ff2e37f55a569d124`.

Independent review parent: https://github.com/amdpilot-org/sglang/pull/3302.

## Result

The candidate's valid API validation, identity derivation, and Kimi K3 image
artifact-cache changes are preserved. The independently reported native
video-only failure was reproduced at the exact candidate and corrected:
`mm_cache_ids` now aligns with native media in image, video, then audio order,
including batched video requests.

Before the correction, the retained regression failed with:

```text
ValueError: mm_cache_ids has 1 entries for 0 images
1 failed, 8 deselected
```

After the correction, the focused candidate and adversarial suite reports:

```text
94 passed, 27 subtests passed
```

The broader OpenAI Chat/Responses serving unit suites additionally report
`170 passed, 70 subtests passed`; these validate compatibility, not a real
multimodal `previous_response_id` no-I/O cache hit.

Raw output is retained at `/job/evidence-j-49b38e785767/`, including
`before/native_video_cache_id.txt` and `after/focused.txt`.

## Scope not claimed as fixed

Source inspection still finds only `processors/kimi_k3.py` calling
`prepare_media_artifacts`. No compatible multimodal model weights were
provided, so there is no honest real-model proof for video/audio no-I/O hot
hits, general image processor coverage, or Responses `previous_response_id`
historical-media reuse. The deterministic tiny Llama fixture cannot validate
those multimodal semantics and was not used as substitute evidence.

One AMD Instinct MI355X was visible through Torch 2.11.0+rocm7.2, but no GPU
model execution was performed. No native source changed, so a native rebuild
was not applicable.

## Reproduction

```bash
/tmp/amdpilot-repo-j-49b38e785767/venv/bin/python -m pytest -q \
  test/registered/unit/managers/test_mm_hashes.py \
  test/registered/unit/entrypoints/openai/test_protocol.py \
  test/registered/unit/parser/test_jinja_template_utils.py \
  test/registered/unit/multimodal/test_media_artifact_processor.py
/tmp/amdpilot-repo-j-49b38e785767/venv/bin/python -m pytest -q \
  test/registered/unit/entrypoints/openai/test_serving_chat.py \
  test/registered/unit/entrypoints/openai/test_serving_responses.py
/tmp/amdpilot-repo-j-49b38e785767/venv/bin/python -m compileall -q \
  python/sglang/srt
git diff --check 358c163250ad3b1f62939b01ce1314a0a31a0365
```
