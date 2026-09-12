# MiniMax-H3 local-path loading investigation

Upstream issue: https://github.com/sgl-project/sglang/issues/33528

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1840

## Outcome

The prepared source already contains the issue-specific fix. No additional source
change is justified.

The screenshot reports:

```text
AttributeError: 'DiffusersTI2VPipelineConfig' object has no attribute 'audio_vae_config'
```

The reproduction used the local path `/workspace/vlm_dataset/MiniMax-H3`. At the
reported revision (`03f44c978acb00ae1ca45deb94e71d000c31b183`), native-pipeline
detection only performed substring matching against organization-qualified names
such as `minimaxai/minimax-h3`. Consequently, that bare local directory did not
select `MiniMaxH3Pipeline`; fallback through the generic Diffusers TI2V config is
consistent with the class named in the traceback.

Upstream PR https://github.com/sgl-project/sglang/pull/33365, merged at
2026-08-04T06:03:35Z, fixed this exact case by matching the final local-directory
name exactly. The screenshot is timestamped 2026-08-04 06:57:07, while its
environment identifies revision `03f44c978`, which predates that merged fix.

In the prepared checkout at base `358c163250ad3b1f62939b01ce1314a0a31a0365`:

- `/workspace/vlm_dataset/MiniMax-H3` resolves to `MiniMaxH3Pipeline`.
- A real local directory named `MiniMax-H3` resolves through `get_model_info` to
  `MiniMaxH3Pipeline` and `MiniMaxH3PipelineConfig`.
- `MiniMaxH3PipelineConfig` defines `audio_vae_config` as a
  `MiniMaxH3AudioVAEConfig`, preventing the reported generic-config attribute
  failure.
- The prepared repository's existing regression for a local MiniMax-H3 path
  passes.
- Independent boundaries `MiniMax-H3.5` and `MiniMax-H3-4B` do not match, while
  a Hugging Face cache path containing `models--MiniMaxAI--MiniMax-H3` still does.

## Evidence

Raw command output is retained beside this report:

- `current-regression.txt`: focused repository regression, 1 passed.
- `routing-boundaries.txt`: reported-revision matcher failure and current matcher
  success for the exact path, plus positive and negative boundary cases.
- `model-info-routing.txt`: actual current registry resolution to the native H3
  pipeline and config.
- `gpu-inventory.txt`: assigned device inventory (one AMD Instinct MI355X,
  gfx950). No GPU kernel was run because the defect is deterministic CPU-side
  path/config selection.

The downloaded issue screenshot and source snapshots used for comparison remain
outside the worktree under `/tmp/amdpilot-repo-j-e653511a10ab/`.

## Limitations

The MiniMax-H3 weights were not available. The assigned machine has one gfx950
GPU, not the reported eight NVIDIA H800 GPUs, so this investigation does not
claim full model loading, generation quality, CUDA behavior, tensor-parallel,
or Ulysses/distributed reproduction. It verifies only the exact pre-load routing
defect identified by the traceback and the current correction.
