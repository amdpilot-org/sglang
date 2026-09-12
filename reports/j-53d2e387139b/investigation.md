# Investigation: hybrid-GDN tracked conv-state dtype

Upstream issue: https://github.com/sgl-project/sglang/issues/31719

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2215

## Findings

The prepared base `358c163250ad3b1f62939b01ce1314a0a31a0365` still performed a
plain indexed assignment from the runtime activation tensor into the
independently configured convolution cache in
`GDNAttnBackend.forward_extend`. PyTorch rejects that operation when the
dtypes differ. The failure was reproduced on CPU and the assigned AMD
Instinct MI350X (`gfx950`) for BF16-cache/FP16-source and
FP32-cache/BF16-source pairs; see `raw/before_dtype_mismatch.txt`.

An upstream search found two relevant open candidates:

- https://github.com/sgl-project/sglang/pull/36289 applies the narrow GDN
  cache-boundary conversion used here. It remained open and review-required
  when inspected on 2026-09-12, and its change was absent from the prepared
  base.
- https://github.com/sgl-project/sglang/pull/30950 changes global conv-cache
  dtype resolution. That is broader than necessary and would remove the
  reported reverse-direction mismatch only by coupling defaults, while an
  explicit cache-layout override can still intentionally differ from runtime
  activations.

The implemented correction preserves `SGLANG_MAMBA_CONV_DTYPE` and converts
only the tracked activation immediately before the indexed cache write.
Regression coverage checks both mismatch directions, an FP32 cache, same-dtype
writes, non-selected slot integrity, and an empty selection.

## Scope limitation

The reported `QuantTrio/Qwen3.6-27B-AWQ` weights and NVIDIA SM120/CUDA runtime
were not available. No full-model serving or semantic-accuracy claim is made.
The gfx950 check validates the implicated tensor write and conversion only.
No native component was changed or rebuilt.
