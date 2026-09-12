# Independent review of PR 1758 at `7a3c6941121615ac6333ccafa9d3873ee4e61a92`

Recommendation: **request changes**. The candidate is a meaningful partial fix, but it does not fully preserve per-output rollout state.

The recorded base was exactly `358c163250ad3b1f62939b01ce1314a0a31a0365`; the image-prepared checkout did not differ. Candidate tests copied outside the checkout reproduced four failures on that base: trajectory tensors collapsed to output 0 and caller-provided raw latents skipped IDs/packing. The grouped residency-manager regression already passed because the base installs the manager in `forward_batch`.

At the exact candidate commit, all seven candidate regressions passed. A real gfx950 check also showed that `rollout_log_probs` and `RolloutDitTrajectory.latents` merge into the expected batch shapes and exactly match independently built CPU tensors.

## Blocking counterexample

`_concat_rollout_trajectory_data` assigns `denoising_env=first.denoising_env`, and `_select_output_rollout_trajectory` returns that environment without slicing. For three singleton outputs tagged with distinct `guidance` and nested `image_kwargs` tensors, all three final results therefore carry output 0's environment. The independent test observed guidance `[1.0, 1.0, 1.0]` rather than `[1.0, 2.0, 3.0]`.

This field is part of `RolloutTrajectoryData`, is collected from each denoising invocation, and the existing HTTP helper recursively slices its batch-indexed tensors. Preserving only the first environment is the same class of silent per-sample corruption described by Bug 1, even though the candidate correctly repairs log probabilities, debug tensors, and DiT latents.

The candidate should merge batch-indexed environment values across expanded outputs and select the corresponding environment for each `GenerationResult`, with coverage for tensor values nested in dictionaries/lists as well as `guidance`.

## Scope and limitations

The loaded source paths were `/job/repo/python/sglang/...`, not an installed older copy. No native rebuild was needed because the diff is Python-only. Tests ran with Torch 2.11.0+rocm7.2 and HIP 7.2.26015 on one AMD Instinct MI355X (`gfx950:sramecc+:xnack-`). No SD3/FLUX.2 weights were available, so this is not a full model, semantic-quality, GRPO-convergence, NVIDIA H20, distributed, or multi-node qualification.

Selected import/GPU evidence and the adversarial test are retained under `evidence/`; complete raw pytest logs were preserved outside the checkout at `/job/review-evidence-j-5fef458e8ae5/` while revisions were switched.
