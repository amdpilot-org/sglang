# Independent review of amdpilot-org/sglang#631

Candidate: `e10d8adc527259f9bfd2a489675120226794f3fc`

Upstream issue: https://github.com/sgl-project/sglang/issues/38574

Mirror issue: https://github.com/amdpilot-org/sglang/issues/655

## Verdict

Request changes. The candidate is a useful error-message improvement and its
new focused tests pass, but it does not resolve the original NEXTN startup
failure. With dense BF16 MTP checkpoint weights, no MTP ignore entry, and a
broad `Linear` W4A8 target, construction still raises
`NotImplementedError`. The new exception makes that failure actionable by
including the full prefix, matched target, format, quantization arguments,
and safe ignore guidance.

This is therefore a partial fix, not a full original-issue fix and not merely
test-only hardening. The candidate's own report accurately acknowledges this
limitation, but the original issue also requests automatic dense draft-layer
handling (or equivalent startup behavior) and documentation.

## Evidence

On prepared base `358c163250ad3b1f62939b01ce1314a0a31a0365`, a real minimal
`CompressedTensorsConfig` with `targets: ["Linear"]`, packed W4 group weights,
and dynamic FP8 token activations reproduced the generic construction error
through `ReplicatedLinear` at prefix
`mtp.layers.0.self_attn.qkv_proj`. Adding `re:mtp\\..*` selected
`UnquantizedLinearMethod` and created BF16 weights.

At the exact candidate commit, the same non-ignored construction still failed,
but its exception contained the module prefix, `Linear` target,
`pack-quantized` format, W4/A8 arguments, `quantization_config.ignore`
guidance, and a warning not to ignore layers carrying `weight_packed` or
`weight_scale`. The candidate regression passed, as did the surrounding 21
compressed-tensors tests.

An independent adversarial check patched
`UnquantizedLinearMethod.create_weights` to fail if the unsupported broadly
targeted MTP layer were silently routed to dense storage. The candidate raised
before selecting that method. This validates safe pre-load scheme selection,
but it cannot prove actual packed-checkpoint loading: checkpoint tensor names
are not available at this construction point.

The explicit-ignore BF16 path ran on one AMD Instinct MI355X (`gfx950`) using
Torch `2.11.0+rocm7.2`/HIP `7.2.26015`; output matched an independent CPU FP32
linear reference after BF16 conversion (maximum absolute error `0.0` for the
small deterministic case). Imports resolved to `/job/repo/python/sglang` and
the reviewed compressed-tensors source in that tree.

## Remaining counterexamples and limitations

- Dense `mtp.layers.0.self_attn.qkv_proj.weight` plus a broad `Linear` W4A8
  target and no MTP ignore entry still fails during construction, so NEXTN
  startup remains broken for the reported llm-compressor checkpoint shape.
- The candidate does not inspect checkpoint tensor names or automatically
  distinguish dense `.weight` from genuine `weight_packed`/`weight_scale`
  storage. Its packed safety test models the pre-load decision, not a complete
  packed checkpoint load.
- No full Qwen3-Next checkpoint was present, so end-to-end server startup on
  the original 2x H100 NVL TP2/EP2 architecture was not run. The reproduction
  exercised the actual constructor and scheme-selection implementation on one
  AMD MI355X instead.
- The candidate adds no user documentation for the MTP ignore requirement.
- No native source changed, `repository-environment.json` reports no separate
  native target, and no native rebuild was applicable.

Raw evidence is retained outside the checkout under
`/job/raw/j-87923e5e2a49/`.
