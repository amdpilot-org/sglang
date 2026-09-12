# Investigation: nested JSON model configuration overrides

Upstream issue: https://github.com/sgl-project/sglang/issues/33505

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1851

The prepared base already contains the source fix reported by the issue. Upstream
PR https://github.com/sgl-project/sglang/pull/33351 merged on 2026-08-03 and
changed `get_config` so a dictionary override targeting an existing
`PretrainedConfig` updates that sub-configuration instead of replacing it with
a plain dictionary.

`raw/shallow_control.txt` demonstrates the old behavior independently: applying
the former top-level `PretrainedConfig.update` to a `text_config` override makes
`text_config` a `dict`, after which attribute access raises `AttributeError`.
`raw/issue_shape_current.txt` applies the issue's nested `rope_parameters` shape
through the checked-out `get_config`; `text_config` remains a `LlamaConfig`, its
unmentioned `max_position_embeddings` and `hidden_size` survive, and the rope
override is present.

The regression was strengthened to cover that issue-shaped override and the
independent scalar/plain-dictionary replacement boundaries. The complete
focused loader suite output is in `raw/pytest_hf_transformers_loading.txt`.

The reported `nvidia/Qwen3.6-27B-NVFP4`, two-GPU CUDA server command was not run:
the model weights and NVIDIA hardware are unavailable, and this job provides a
single AMD gfx950 GPU. This configuration-loading defect is exercised before
weight loading and does not require GPU execution. Consequently this report
does not claim model semantic accuracy, NVFP4 support, server startup, or a TP=2
reproduction.
