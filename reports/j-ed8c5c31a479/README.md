# Investigation evidence

Upstream issue: https://github.com/sgl-project/sglang/issues/32169

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2116

The prepared base reproduced the issue with the real
`OpenGVLab/InternVL2_5-2B` processor files: Hugging Face returned a bare
`CLIPImageProcessor`, and SGLang attempted an unguarded `.tokenizer` access.

The change loads the tokenizer separately only for a resolved object that is
neither a `PreTrainedTokenizerBase` nor an object already exposing
`.tokenizer`, then attaches it for downstream callers. Tests cover that
regression and both boundary paths.

Raw evidence is retained in `raw/`. The post-change real-model run passed the
original failure point, but the prepared SentencePiece 0.2.2 dependency then
rejected the model's SHA-256-matching tokenizer file because vocabulary piece
354 is a literal NUL. Therefore this report does not claim full server startup,
model inference, semantic correctness, or GPU validation.

Relevant source paths:

- `python/sglang/srt/utils/hf_transformers/processor.py`
- `test/registered/unit/utils/test_hf_transformers.py`

No native library applies to this Python-only change, and none was rebuilt.
