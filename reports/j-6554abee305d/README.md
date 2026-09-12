# Investigation evidence

- Upstream issue: https://github.com/sgl-project/sglang/issues/31248
- Mirror issue: https://github.com/amdpilot-org/sglang/issues/2785
- Base: `358c163250ad3b1f62939b01ce1314a0a31a0365`
- Reported model config fetched from:
  `https://huggingface.co/unsloth/gemma-4-26B-A4B-it-NVFP4/resolve/main/config.json`
- Assigned GPU: AMD Instinct MI350X, gfx950, ROCm 7.2.
- Runtime evidence directory: `/tmp/amdpilot-repo-j-6554abee305d/`
- Focused pytest log:
  `/tmp/amdpilot-repo-j-6554abee305d/logs/pytest-mixed-precision.log`
- Current model config selection log:
  `/tmp/amdpilot-repo-j-6554abee305d/logs/current-model-config-selection.log`
- Source issue snapshots:
  `/tmp/amdpilot-repo-j-6554abee305d/upstream-issue.json` and
  `/tmp/amdpilot-repo-j-6554abee305d/mirror-issue.json`
- Historical reference checkout (outside the job worktree):
  `/tmp/amdpilot-repo-j-6554abee305d/vllm-src`; vLLM commit
  `af9b69f977bd1166ed63c46f9ccbd3a02344ae4f` removed Sparse-Marlin 24.

The current model config has `sparsity_config: {}`. It describes mixed FP8
attention and NVFP4 MLP groups, and explicitly ignores the vision tower linear
projections implicated by the issue traceback. Consequently, the reported
current checkpoint no longer reproduces the exception during scheme selection.

This contribution does not claim implementation of W4A16 2:4 sparsity. Such an
implementation needs a maintained native sparse kernel, NVIDIA compilation,
and independent GPU numerical validation, none of which is available on the
assigned AMD node.
