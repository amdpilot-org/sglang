# Independent review of PR 2382 at `4af0a2a8dbba75d138607e3268e1f0006550025a`

Upstream issue: https://github.com/sgl-project/sglang/issues/31018

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/2333

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2421

Candidate PR: https://github.com/amdpilot-org/sglang/pull/2382

## Verdict

Recommendation: **accept**. The candidate fully resolves the original source-level runtime-gamma mismatch. This is a functional fix with a failing-before/passing-after regression, not test-only hardening.

At the recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`, the real `DSparkVerifyPlanner.compute_confidence_tensor` passed a confidence tap containing `batch * runtime_gamma * hidden_width` elements to `DeepseekV4ForCausalLMDSpark.compute_confidence`, which reshaped it with the checkpoint gamma. With checkpoint gamma 5 and runtime gamma 7, the reproduced exception was:

```text
RuntimeError: shape '[2, 5, -1]' is invalid for input of size 56
```

At exact candidate commit `4af0a2a8dbba75d138607e3268e1f0006550025a`, the planner explicitly passes `self.gamma`, and the model uses that value for both the confidence-tap reshape and Markov-token slice. The same reproduction then produced confidence hidden shape `(2, 7, 4)` and Markov shape `(2, 7, 2)` with exit code 0.

The two source changes are semantically identical to the open upstream proposed fix in https://github.com/sgl-project/sglang/pull/31016. No other model defines a `compute_confidence` hook in the reviewed tree, so adding the keyword at this call site does not break another in-tree hook implementation.

## Evidence

- Prepared checkout was exactly the recorded base before testing; no difference from the image-prepared checkout was observed.
- Imports resolved to `/job/repo/python/sglang/srt/models/deepseek_v4_dspark.py` and `/job/repo/python/sglang/srt/speculative/dspark_components/dspark_planner.py` at both revisions.
- Running the candidate's regression file against the base gave `1 failed, 2 passed`; the override test failed at the issue's `[2, 5, -1]` reshape.
- Running it at the candidate gave `3 passed`.
- The candidate's GPU numerical check passed on AMD Instinct MI355X with checkpoint gamma 5, runtime gamma 7, output `(2, 7)`, and maximum absolute error `0.0`.
- Independent GPU cases passed for runtime gamma larger than checkpoint `(5 -> 7)`, smaller than checkpoint `(7 -> 3)`, the gamma-1 boundary, Markov enabled and disabled, and checkpoint-default fallback when no runtime gamma is supplied.
- `git diff --check` found whitespace in candidate-owned captured pytest output files. This is cosmetic report-output whitespace, not a source or behavior defect, and is not grounds to reject the functional fix.

Raw command output and fetched issue/PR metadata were preserved outside the revision-switched checkout at `/job/review-evidence-j-81c16a2014cc/` during review.

## Limitations

The assigned environment has one AMD Instinct MI355X (`gfx950`) with PyTorch `2.11.0+rocm7.2` and ROCm/HIP `7.2.26015`. It does not have the report's eight NVIDIA H20 GPUs, CUDA 12.9 environment, DeepSeek-V4-Flash-DSpark weights, or a 4-way TP/DP setup. Therefore I did not reproduce full server startup or CUDA graph capture and do not claim NVIDIA compiler/ISA, full-model, semantic-quality, distributed, or multi-node validation. The deterministic direct exercise covers the exact planner/model contract that caused capture to fail, and real GPU tensors verify its numerical data path on the assigned architecture.

No C++ or other native source changed. Native rebuild was therefore not applicable; the reviewed Python imports came from the candidate checkout rather than an installed wheel.

