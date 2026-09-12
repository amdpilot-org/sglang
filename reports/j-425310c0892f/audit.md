# DeepSeek-V4-Pro defaults audit

Environment: one AMD Instinct MI355X reported by `rocm-smi` as `gfx950`, with
PyTorch 2.11.0+rocm7.2. Raw command output is retained in `raw/`.

## Reproduced baseline

At base commit `358c163250ad3b1f62939b01ce1314a0a31a0365`:

- `docker/rocm.Dockerfile` exported `SGLANG_USE_ROCM700A=1` in the shared final
  stage, so every ROCm image inherited it, including `gfx950-rocm720`.
- `OpenAIServingChat._apply_jinja_template` used the unset
  `SGLANG_DEFAULT_THINKING` default (`False`) for every encoder.
- The DSV4 branch used `SGLANG_DSV4_REASONING_EFFORT` only when nonempty;
  otherwise it passed `None`, selecting the encoder profile's default.

The added regression fails on that baseline because it expects DSV4-only
thinking/high defaults and a `0` image default for the reported stage.

## What the variables control

- `SGLANG_DEFAULT_THINKING` selects `ThinkingMode.THINKING` versus
  `ThinkingMode.CHAT` while rendering chat input. It is not hardware-specific.
- `SGLANG_DSV4_REASONING_EFFORT` is read only by the DSV4 encoder path when a
  request did not supply `reasoning_effort`; accepted values depend on the
  detected checkpoint encoder profile.
- `SGLANG_USE_ROCM700A` is read at import time by `dp_attention.py`. It selects
  `SUM_LEN` rather than `MAX_LEN` as the default DP-attention CUDA-graph padding
  mode for a documented ROCm 7.0.0-alpha RCCL workaround. `ROCM700A` is a
  software-version label, not the `gfx950` ISA name. On the actual GPU, separate
  processes demonstrated `0 -> MAX_LEN` and `1 -> SUM_LEN`.

## Scope and limitations

The semantic defaults are applied only when the resolved chat encoder is
`dsv4`. Request values and explicitly set environment variables retain
precedence, including explicit false/empty values. Other encoders retain chat
mode by default. The image change is limited to `gfx950-rocm720`; all other
existing image stages retain the former value `1`.

DeepSeek-V4-Pro weights were unavailable, so no full-model quality, semantic
accuracy, distributed execution, or throughput claim is made. The GPU
calculation in `raw/gpu-and-flag-evidence.log` verifies execution on the assigned
MI355X and agrees with the independent CPU result; it is environment evidence,
not evidence of DeepSeek-V4-Pro semantics.
