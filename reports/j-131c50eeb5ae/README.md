# DeepSeek-V4 configuration dispatch investigation

Upstream issue: https://github.com/sgl-project/sglang/issues/33207

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1928

## Finding

The prepared source already contains the relevant solution. A DeepSeek-V4
checkpoint is parsed as `_DeepseekV4ConfigAlias`, but SGLang's automatic model
dispatch recognizes `DeepseekV4ForCausalLM` and selects the native
`sglang.srt.models.deepseek_v4.DeepseekV4ForCausalLM` implementation before the
Transformers fallback is considered.

The exact reported exception remains reproducible if
`transformers.AutoModelForCausalLM.from_config()` is called directly with the
alias. That forced call is not the current automatic SGLang serving path, and
registering the alias as a Hugging Face V3 causal-LM config would incorrectly
route a V4 checkpoint to a different model implementation.

The native DeepSeek-V4 implementation and config alias were introduced by the
merged upstream DeepSeek V4 work in https://github.com/sgl-project/sglang/pull/23882.

## Evidence

- The real `DeepSeek-V4-Flash-0731` `config.json` was downloaded without model
  weights to `/tmp/amdpilot-repo-j-131c50eeb5ae/model-config/config.json`.
- `/tmp/amdpilot-repo-j-131c50eeb5ae/evidence/current-config-dispatch.txt`
  records successful construction of `ModelConfig`, native architecture
  resolution, and the intentionally forced Hugging Face boundary exception.
- `/tmp/amdpilot-repo-j-131c50eeb5ae/evidence/pytest-deepseek-v4-dispatch-passing.txt`
  records the focused regression and independent unknown-architecture fallback
  boundary passing.
- `/tmp/amdpilot-repo-j-131c50eeb5ae/issue-33207.json`,
  `/tmp/amdpilot-repo-j-131c50eeb5ae/issue-33207-comments.json`, and
  `/tmp/amdpilot-repo-j-131c50eeb5ae/issue-33207-timeline.json` retain the issue
  state inspected during the investigation.
- `/tmp/amdpilot-repo-j-131c50eeb5ae/evidence/pr-23882.json` retains the related
  merged change metadata.

## Limitations

The assigned accelerator is one AMD Instinct MI355X (`gfx950`), not the two
NVIDIA B200 or RTX PRO 6000 Blackwell GPUs in the report. The full checkpoint
weights were not available or downloaded. Consequently this investigation does
not claim full-model startup, tensor-parallel execution, FlashInfer MXFP4
execution, NVIDIA compiler/ISA validation, output accuracy, or multi-GPU
reproduction. GPU execution was unnecessary for the configuration-dispatch bug
and was not used as evidence.
