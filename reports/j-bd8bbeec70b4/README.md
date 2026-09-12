# Independent review of PR 1647

Reviewed https://github.com/amdpilot-org/sglang/pull/1647 at exact commit `4896853f26a6f6571d82aee3d965db71e6ec9fac` against upstream issue https://github.com/sgl-project/sglang/issues/35324 and mirror issue https://github.com/amdpilot-org/sglang/issues/1681. The recorded base was `358c163250ad3b1f62939b01ce1314a0a31a0365`; the image-prepared checkout matched it.

## Recommendation

**Accept as a narrow, partial containment; it does not fully resolve the original issue.**

The base recovery examined only probability column 0 for NaN. Independent CPU evidence reproduced the four inherited counterexamples (NaN outside column 0, positive infinity, a negative element, and a zero-sum row), plus negative infinity. On the assigned GPU, the base NaN case reached `torch.multinomial`, raised its device-side assertion, and left the process with an HSA launch failure. This is the same concrete failure class as the issue, but not the original model/workload reproduction.

At the exact candidate commit, the candidate suite passed (`4 passed`, including CPU and GPU subtests). An independent probe confirmed on CPU and one gfx950 GPU that NaN in either position, positive/negative infinity, a negative value, and zero mass all recover to token-0 one-hot rows; synchronized `torch.multinomial` then succeeds. Valid normalized and unnormalized rows remain unchanged. The imported implementation was `/job/repo/python/sglang/srt/speculative/dspark_components/dspark_draft.py`, so the tests exercised the checked-out source rather than an installed copy.

The candidate changes Python only. No native source or build metadata changed, so no native rebuild was required or applicable.

## Why this is not a full original-issue fix

The evidence establishes containment only at the eager DSPARK `torch.multinomial` boundary. The current prepared source defaults `SGLANG_DSPARK_FAST_SAMPLING` to true and also has a folded sampling path; neither is changed by the candidate. The issue's CUDA traceback makes the eager multinomial boundary relevant, but the rare producer of malformed probabilities was not identified. DeepSeek-V4-Flash-0731 weights, eight NVIDIA H100 GPUs, CUDA 13, TP=8, the reported server configuration, and week-scale production traffic were unavailable. The assigned device was one AMD Instinct MI350X (`gfx950:sramecc+:xnack-`) under ROCm 7.2.

One observability limitation also remains: `_DRAFT_PROBS` still declares `NotNaN()`, so recovery handles infinity, negative values, and zero-sum rows, but invariant signaling/telemetry detects only NaNs. This does not defeat containment, but it does not help identify all malformed-row producers.

Raw commands and outputs are retained under `raw/`, including the base device assertion, exact candidate pytest output, source import path, environment, and independent probes.
