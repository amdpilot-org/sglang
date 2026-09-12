# Independent review of PR 1908

Reviewed candidate: https://github.com/amdpilot-org/sglang/pull/1908 at exact commit `ac20ceff67d42304ddaeb80b1800097b9c5b4cce`.

Original issue: https://github.com/sgl-project/sglang/issues/34000

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1942

## Recommendation

Accept for the original issue. The candidate fully resolves the three reported failures under the issue's stated contract (`num_outputs_per_prompt > 1`, `rollout=True`): per-output rollout tensors and denoising environments remain distinct, grouped forward has a residency manager, and raw caller-provided latents receive latent IDs and packing. This is a source fix with regression coverage, not test-only hardening.

One adjacent counterexample remains: denoising-environment capture without rollout log-probabilities is merged but not selected per output. `_select_output_rollout_trajectory` derives `batch_size` only from `rollout_log_probs`, defaults it to 1 when log-probabilities are absent, and consequently returns the complete K-row environment to every result. That mode is outside the original report's explicit `rollout=True` contract, where log-probabilities are collected, so it does not change the original-issue verdict. It should be handled in a follow-up or by validating that environment capture requires rollout.

## Evidence

The prepared branch was exactly `358c163250ad3b1f62939b01ce1314a0a31a0365`, matching the recorded base. Candidate metadata was fetched from the mirror and the exact commit was checked out detached. Raw logs and the exact candidate diff are retained outside the checkout in `/job/review-evidence/j-cb0c035ba265/`.

On the base, the candidate rollout regression failed because the merged log-probability shape was `(1, 4)`, not `(3, 4)`. The raw-latent regression also failed: `latent_ids` remained `None` and latents remained `(1, 4, 8, 8)` rather than packed `(1, 64, 4)`. The grouped-forward residency test already passed on this newer prepared base, so Bug 2 was pre-fixed relative to the issue snapshot.

On the exact candidate, the focused suite passed 41 tests. An independent GPU fixture constructed three singleton rollout trajectories tagged 1, 2, and 3 and exercised the real merge and result-selection functions. Log-probabilities, guidance, tensors nested in `image_kwargs`, `pos_cond_kwargs`, and `neg_cond_kwargs`, and DiT latents all returned tags `[1, 2, 3]` and matched direct numerical references.

The adversarial environment-only fixture (no `rollout_log_probs`) failed: each result contained guidance `[1, 2, 3]` instead of its own singleton value. This is the remaining counterexample recorded in `result.json`.

## Environment and limitations

Imports resolved to `/job/repo/python/sglang/...`, while Torch resolved to `/opt/venv/lib/python3.12/site-packages/torch`. The interpreter reported Torch `2.11.0+rocm7.2`, HIP `7.2.26015`, one AMD Instinct MI355X, and device capability `(9, 5)` (`gfx950`). GPU execution was limited to deterministic tensor operations through the actual merge/selection source.

No SD3 or FLUX.2 weights were available. This review therefore does not claim a full diffusion server run, semantic image quality, GRPO convergence, NVIDIA H20 equivalence, or distributed/multi-node behavior. The provided-latent check uses the actual stage with a packed-config stub. The candidate changes only Python; no native source changed and no native rebuild was applicable.
