# Investigation evidence

The reported failure was an argument-routing failure: a bare local directory named
`MiniMax-H3` was not recognized as diffusion, so `sglang serve` sent diffusion-only
arguments to the LLM parser.

Current related fix: https://github.com/sgl-project/sglang/pull/33365 was merged on
2026-08-04 as commit `101bb2327cdebf310d5261775f00eaef13a2e168`. Its stated
problem and fix match the issue exactly: recognize `/data/models/MiniMax-H3` as a
native diffusion model and resolve it to `MiniMaxH3Pipeline`.

Prepared source at `358c163250ad3b1f62939b01ce1314a0a31a0365` contains that
behavior. The issue-specific offline probe produced:

```text
selected_backend: diffusion
remaining_argv: []
num_gpus: 2
tp_size: 2
ulysses_degree: 1
performance_mode: memory
layerwise_offload_components: ['dit,text_encoder,vae']
dit_offload_prefetch_size: 1.0
dit_layerwise_resident_layers: 20.0
enable_torch_compile: False
model_variant: fl2va
pipeline: MiniMaxH3Pipeline
exact_detected: True
boundary: /mnt/models/MiniMax-H3.5 True None
boundary: /mnt/models/MiniMax-H3-4B True None
boundary: /mnt/models/not-MiniMax-H3 True None
all_assertions_passed: true
```

The broad registry detector recognizes the near-name boundaries as diffusion, but
the non-diffusers pipeline resolver correctly does not map them to MiniMax-H3. That
does not reproduce the reported LLM-parser rejection for the exact path and is
recorded here rather than expanded into an unrelated change.

Focused existing tests produced `3 passed, 17 warnings in 2.05s`.

The first ad-hoc boundary probe exited 1 after all exact-path output because the
test harness attempted to patch `sglang.cli.utils.hf_hub_download`, which is lazily
imported and therefore is not a module attribute. The corrected offline probe above
did not use that invalid patch and exited 0.

GPU inventory showed Torch `2.11.0+rocm7.2`, HIP `7.2.26015`, one visible device,
and `AMD Instinct MI350X`. No GPU model execution was performed.
