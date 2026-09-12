# Independent review of PR 1465

Upstream issue: https://github.com/sgl-project/sglang/issues/35324

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1499

Candidate: https://github.com/amdpilot-org/sglang/pull/1465 at `1dd7ea6afbbe9cdac3c31585526b779869bb9e3b`

Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

## Recommendation

Request changes. The candidate is test-only hardening, not an original-issue fix. Its three tests pass unchanged against the recorded base, because the production recovery was already present there. The candidate therefore provides no failing-before/passing-after regression for its own diff and cannot establish that the intermittent CUDA/H100/TP=8 failure is fully resolved.

The existing base recovery is useful and was verified through the real eager `sample_draft_block` callback on the assigned gfx950 GPU: an all-`-inf` logit row becomes an all-NaN softmax row, is replaced by token-0 one-hot probabilities, and samples token 0. A separate safe GPU probe also verified the all-`-inf` and positive-`inf` softmax cases and compared a valid row with a float64 CPU softmax reference (maximum absolute error `2.20e-08`).

However, `_one_hot_token0` identifies a bad row only through `torch.isnan(probs[:, :1])`, while `torch.multinomial` requires every value to be finite and nonnegative and each row to have positive mass. Independent CPU cases containing a NaN outside column 0, positive infinity, a negative value, or zero total mass all pass through the recovery unchanged and fail multinomial. These direct probability tensors are not ordinary outputs of the immediately preceding PyTorch softmax, so they do not disprove containment of the demonstrated all-NaN softmax mechanism; they do show that the candidate has not established the broader failure cause encoded by the original generic CUDA assertion.

An isolated gfx950 adversarial process passed unrecovered invalid probabilities to HIP `torch.multinomial` and aborted with `HSA_STATUS_ERROR_EXCEPTION` / `CUDA error: unspecified launch failure`, reproducing the device-side failure class. The process-local device failure did not prevent later clean-process GPU probes.

## Candidate-specific findings

- No production or native source is changed by the candidate; only a test and report artifacts are added.
- The candidate test passes at both the candidate commit and the recorded base (`3 passed` each).
- The candidate test calls `expect(_DRAFT_PROBS, probabilities)` directly rather than the production `sample_draft_block` path.
- The existing recovery entered the source in earlier commit `a31542ebd9` and is byte-identical between the recorded base and candidate.
- The candidate PR body cites mirror issue `1393`, not the assigned mirror issue `1499`.
- The candidate accurately demonstrates containment for all-NaN softmax rows, but its `candidate_verified` outcome is too broad for the original intermittent production issue.

## Architecture and environment limitations

Testing used one AMD Instinct MI355X (`gfx950`, exposed by PyTorch as `cuda:0`) with PyTorch `2.11.0+rocm7.2` and HIP `7.2.26015`. The original environment used eight NVIDIA H100-class GPUs, CUDA, TP=8, SGLang 0.5.16, and unavailable DeepSeek-V4-Flash-0731 weights. The full server path, fast DSPARK sampling kernel, distributed synchronization, model semantics, week-long production traffic, and the upstream source of the rare invalid probabilities were not reproduced. No native code changed, so no native rebuild was applicable.

Raw salient outputs are retained in `raw/`; the larger process-abort log and executable probes remain outside the checkout under `/job/review-evidence-j-3e09cbbbc4d1/` while revisions were switched.
