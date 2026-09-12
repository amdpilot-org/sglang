# FLUX.2 explicit SageAttention backend investigation

Upstream issue: https://github.com/sgl-project/sglang/issues/39070

Mirror issue: https://github.com/amdpilot-org/sglang/issues/738

Base commit: `358c163250ad3b1f62939b01ce1314a0a31a0365`

## Finding

The reported model-whitelist rejection is already fixed at the prepared base.
Upstream PR [#37441](https://github.com/sgl-project/sglang/pull/37441), merged as
`9175590aa06186379d9cda555d1a1a31eab889a7` on 2026-09-02, changed
`supported_attention_backends` from a strict allowlist into an automatic-selection
constraint. An explicitly requested backend now bypasses membership validation but
still must resolve on the current platform and satisfy the layer's semantic
requirements.

FLUX.2 still omits `SAGE_ATTN` from
`Flux2Transformer2DModel._supported_attention_backends`. The new regression uses
that exact set and proves:

1. An explicit `sage_attn` request is admitted after platform resolution.
2. An implicit `sage_attn` preference still falls back to a backend in FLUX.2's
   automatic-selection set.
3. An explicit backend still fails closed when it lacks a required capability.

Restoring the pre-#37441 membership check makes the issue-specific regression fail
with the reported error. The raw failure is in
`raw/pre_37441_flux2_selector.txt`; the passing current-base runs are in
`raw/current_flux2_selector.txt` and `raw/current_full_selector.txt`.

## Hardware boundary

The assigned device is one AMD Instinct MI350X (`gfx950`) under ROCm 7.2. The real
ROCm platform resolver raises `ValueError: SAGE_ATTN is not supported on rocm.`;
see `raw/gpu_platform_probe.txt`. Therefore no SageAttention kernel or FLUX.2 model
was executed on GPU. This environment cannot qualify the reporter's CUDA Blackwell
backend, image accuracy, or full-model startup, and the model weights were not
available. The regression validates only the reported selector/model-whitelist
interaction.
