# Investigation evidence

Base commit: `358c163250ad3b1f62939b01ce1314a0a31a0365`

The source issue and its three proposed upstream changes were inspected before
editing. The public review refs used were:

- `sgl-project/sglang#34416` at `d1a796422fb146c24c68fc3ceeea45879318b12d`
- `sgl-project/sglang#34417` at `166f1fb3f12bd8065199c3c1aa5e9f6812226587`
- `sgl-project/sglang#34418` at `4c0534b0a76f2876cd8bc5c85bfc2779189ed570`

Bug 1 reproduced in the checked-out `GPUWorker._merge_expanded_output_batches`:
three distinct `[1, 4]` rollout-log-prob rows were collapsed to the first row,
and a partially populated group was mislabeled as a complete trajectory. The
patch concatenates aligned rollout tensors to `[K, ...]`; DiffGenerator then
selects the requested row while retaining a batch dimension of one.

Bug 2 was already fixed in the prepared source. `ComposedPipelineBase.forward_batch`
assigns `get_global_component_residency_manager(...)` to both the pipeline and
executor before `execute_group_with_profiling`. The added regression observes
the manager at grouped execution time and passes on the unmodified base logic.

Bug 3 reproduced in `LatentPreparationStage.forward`: caller-provided raw
`[B,C,H,W]` latents skipped latent-ID construction and packing. The patch runs
the same preparation used for drawn noise when the supplied tensor matches the
stage's raw latent shape. Already-packed tensors remain untouched, preventing a
second shape-compatible but semantically corrupting reshape.

Raw command output is retained in `reports/j-ff98d30c0ca9/raw/`. The GPU check
used the prepared Torch/ROCm environment and device 0, reported as AMD Instinct
MI355X (`gfx950:sramecc+:xnack-`), and compared GPU output against independently
constructed CPU tensors with zero tolerance.

No diffusion model weights were available. No full SD3/FLUX.2 server, semantic
image-quality, GRPO training, distributed, or multi-node claim is made.
