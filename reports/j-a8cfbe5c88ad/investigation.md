# Diffusion multi-output rollout correction

This correction is based on candidate PR https://github.com/amdpilot-org/sglang/pull/1758 at exact commit `7a3c6941121615ac6333ccafa9d3873ee4e61a92` and independent review PR https://github.com/amdpilot-org/sglang/pull/1849.

The candidate was checked out exactly before any correction. A K=3 fixture
gave each singleton trajectory distinct guidance and batch-indexed tensors in
all three conditioning dictionaries, including list and tuple nesting. After
merge and per-result selection, every field was `[1, 1, 1]`; the raw failure is
retained in `evidence/candidate-counterexample.txt`.

The candidate's valid fixes are preserved: rollout log probabilities, debug
tensors and DiT latents are concatenated and selected per output; incomplete
trajectory groups are dropped; raw provided latents receive IDs and packing;
already-packed latents remain unchanged. The recorded base already contained
the grouped residency-manager installation, and its regression remains.

The correction recursively concatenates singleton-batch tensors in
`RolloutDenoisingEnv` while retaining request-shared metadata, and recursively
selects only values whose leading dimension equals the merged output count.
The special batch-list representation of `img_shapes` is also merged and
selected. Tests cover guidance and tensors nested under `image_kwargs`,
`pos_cond_kwargs`, and `neg_cond_kwargs`, plus the prior boundary cases.

No SD3 or FLUX.2 weights were available. Therefore this does not claim a full
diffusion-server, image-semantic, GRPO convergence, NVIDIA H20, distributed,
multi-GPU, or multi-node qualification. The packed-latent behavior is tested
against the actual stage with a config stub, not a loaded FLUX.2 model.
