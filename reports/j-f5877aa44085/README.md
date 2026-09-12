# Investigation report: Qwen3.8/Qwen3.5 NVFP4 repetition

Upstream issue: https://github.com/sgl-project/sglang/issues/35723

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1313

## Outcome

The original serving failure could not be reproduced or qualified in this
environment. The reported checkpoint is private/gated (its `config.json`
request returned HTTP 401) and is an NVIDIA NVFP4 checkpoint, while the
assigned accelerator is one AMD Instinct MI350X (`gfx950`). No source change is
justified without an execution-level reproduction of that architecture and
quantization combination.

The comparison did identify an important versioning detail: `v0.5.17` predates
the reporter's known-good commit `a4ffb996d`. There are 863 commits between
them. That interval includes Qwen3.8-27B model support (`8a1e6e4e4`, PR
#34859), fixes for padded GDN state slots (`955704544`), Qwen3.5 NVFP4 draft
quantization (`03cf2de2e`), and the known-good commit itself, which restores
Triton GDN prefill under deterministic inference (`a4ffb996d`, PR #35632).
Consequently, the report does not establish a single regression introduced by
0.5.17; its claimed working revision also contains later model support and
multiple GDN/quantization corrections.

Current source already contains the deterministic-GDN correction and its
regression tests. The focused policy suite passed 15 tests and 24 subtests.
On the assigned `gfx950`, the Triton GDN suite passed 15 projected
extend/decode subtests and five speculative verification cases against its
independent pure-PyTorch gated-delta recurrence reference. The ROCm suite
explicitly skips the piecewise-CUDA-graph split-op runner because that runner
is not wired on ROCm.

These checks qualify the current backend policy and Triton GDN numerical path
only. They do not qualify the private Qwen3.8-27B NVFP4 checkpoint, semantic
generation quality, FlashInfer on NVIDIA, or a complete HTTP serving run.

## Evidence

- `raw/pr35632.json`: upstream fix description and changed files.
- `raw/fix-35632.diff`: exact known-good commit patch.
- `raw/current-vs-v0517-policy.txt`: deterministic guard present now and
  absent from v0.5.17.
- `raw/all-tag-to-known-good-commits-head.txt` and
  `raw/candidate-fix-commits.txt`: revision comparison.
- `raw/test-policy.log`: focused backend-policy regression output.
- `raw/test-gdn-gfx950.log`: GPU numerical and speculative-boundary output.
- `raw/test-gdn-skip-reason.log`: exact ROCm-only skipped boundary.
- `raw/gpu-environment.txt`: assigned GPU and Torch/ROCm evidence.
- `raw/checkpoint-access.log`: checkpoint metadata HTTP 401 and exit status.
