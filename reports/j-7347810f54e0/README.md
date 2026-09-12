# Investigation of sglang#31243

Upstream issue: https://github.com/sgl-project/sglang/issues/31243

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2271

## Result

The prepared source at `358c163250ad3b1f62939b01ce1314a0a31a0365`
already contains the issue-specific fix. No duplicate runtime change is justified.

The report's command sets `--mm-attention-backend fa4`, which selects the vision
encoder backend, but does not set the language-model `--attention-backend`.
On the reported SM100/SM103 path, the old automatic selection chose
`trtllm_mha`. MiMo-V2.5 has asymmetric K/V widths (`head_dim=192`,
`v_head_dim=128`), while that backend requires equal K/V widths. This accounts
for the reported 3:2 cache-size mismatch during graph capture.

Upstream PR https://github.com/sgl-project/sglang/pull/32818, merged as
`e5c46ff07d7859ceee9d2e59cb03e1bc3ecff118`, explicitly identifies issue
#31243 and added the capability-based correction now present in
`python/sglang/srt/arg_groups/model_override_base.py`: asymmetric-K/V MHA
models resolve to `fa4` on SM100 rather than `trtllm_mha`. The current MiMo
cookbook also distinguishes `--attention-backend fa4` from
`--mm-attention-backend fa4`.

## Verification

- A focused resolver probe simulated the SM100 selection branch. MiMo-style
  asymmetric KV selected `fa4`; a symmetric control selected `trtllm_mha`; an
  asymmetric top-k=2 speculative case selected the normal `flashinfer`
  fallback. All three assertions passed. Raw output:
  `raw/test_sm100_backend_resolution.txt`.
- The existing MiMo override tests passed: 3 passed, 91 deselected. Raw output:
  `raw/test_mimo_model_overrides.txt`.
- On the assigned AMD Instinct MI355X (`gfx950`), the device-side asymmetric MHA
  pool suite passed: 8 tests and 9 subtests. It covers the exact 192/128 MiMo
  dimensions, reversed 128/192 dimensions, a wider 512/256 case, symmetric
  128/128 behavior, SWA overrides, untouched-slot integrity, and the
  prefix-valid guard. Raw output: `raw/test_asymmetric_mha_pool.txt`.
- The full model-override file was also run. It had 90 passes and four failures
  unrelated to MiMo, caused by CUDA-specific expectations while executing on
  ROCm (including rejection of `modelopt_fp4` on ROCm). This run is retained in
  `raw/test_model_overrides.txt` and is not claimed as passing.

## Limitations

The XiaomiMiMo/MiMo-V2.5 weights were not available. The assigned machine has
one AMD gfx950 GPU, not four NVIDIA L20D/B300-class SM100/SM103 GPUs. Therefore
the original TP=4 CUDA graph launch, full model execution, CUDA kernel, and
model accuracy were not reproduced here. The GPU test validates the relevant
asymmetric cache storage path on ROCm only; the resolver probe validates the
SM100 dispatch decision deterministically without claiming SM100 execution.

