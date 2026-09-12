# Issue 31240 investigation

The prepared base already contains the reported correction in
`python/sglang/srt/utils/hf_transformers/processor.py`: when `model_name` is
present, `get_processor()` resolves it with `resolve_runai_obj_uri()` before
passing it to `AutoConfig.from_pretrained()`.

The base also contains the focused regression
`TestGetProcessor.test_resolves_model_name_before_loading_config`. A controlled
failing-before check removed only the two-line `model_name` resolution block:
the test then showed that `AutoConfig` received the raw `s3://bucket/model`
instead of `/cache/model`. Restoring the base implementation made the same test
pass.

The adjacent RunAI utility tests independently cover S3, Google Storage, Azure,
uppercase schemes, local and relative paths, Hugging Face repository IDs,
unsupported URL schemes, `pathlib.Path`, and cache-path determinism. Raw test
outputs and exit codes are retained under `raw/`.

This is a pre-model-load CPU routing defect, so GPU execution would not add
issue-specific evidence. No object-store credentials/endpoint or Qwen3.5
weights were supplied; consequently this investigation does not claim a full
RunAI streamer download, multimodal server startup, or semantic inference.

Upstream issue: https://github.com/sgl-project/sglang/issues/31240

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2270
