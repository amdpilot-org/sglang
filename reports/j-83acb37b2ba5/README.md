# gfx942 fused SiLU post-quant reuse investigation

This report covers the buffer-reuse and output-scale fidelity follow-up to
amdpilot-org/sglang issue 208. It intentionally does not repeat the launch-bound
work already delivered by open mirror pull request 285.

## Result

- The supported Triton fused SiLU-and-mul FP8 path preserved exact per-128-group
  output scales when the same output and scale storage was reused for masked
  batch changes `1 -> 3 -> 5 -> 7`.
- Every reuse case passed the existing numerical gates: scale
  `rtol=5e-3, atol=1e-6`, and dequantized error
  `abs(error) <= actual_scale * 17.0 + 1e-4`.
- All valid output bytes matched an independent CPU reference. The finite input
  matrix included zero, tiny, positive, negative, and larger magnitudes.
- An input/output alias probe also passed those numerical gates, but aliasing is
  not documented as part of the operation contract. This result does not promote
  aliasing to a supported contract.
- The checkout's CUDA-only JIT wrapper remains unsupported on this qualified
  ROCm 7.2/gfx942 stack because `cuda_fp8.h` is unavailable to `hipcc`. The
  exact failure is recorded in `gfx942-results.json`; no toolchain was replaced.
- No production-code change was made because no supported-contract mismatch was
  demonstrated.

## Reproduce

Run from the repository root on one assigned MI300X:

```bash
PYTHONPATH="$PWD/python" \
TRITON_CACHE_DIR=/tmp/sglang-cache-j-83acb37b2ba5/checkout-triton \
SGLANG_JIT_CACHE_DIR=/tmp/sglang-cache-j-83acb37b2ba5/checkout-jit \
/opt/venv/bin/python reports/j-83acb37b2ba5/reproduce_gfx942.py
```

The script performs one bounded JIT compile probe, four real Triton invocations,
and one alias probe. It uses one CUDA event pair per invocation and no warmup,
sleep, unbounded loop, or occupancy stress. Raw values are written to
`gfx942-results.json`.

## Limitations

- This is a kernel-level investigation, not an end-to-end model or DP-attention
  run; no model weights were downloaded.
- The first reuse timing includes one-time Triton compilation. Later timings use
  the cached kernel.
- The alias result is observational only because the kernel does not document or
  promise input/output aliasing.
- Upstream issue sgl-project/sglang 32065 and open upstream pull request 32066
  were read only. No upstream issue, pull request, or comment was modified.
