# Independent review of amdpilot-org/sglang PR 752

Candidate: https://github.com/amdpilot-org/sglang/pull/752  
Exact commit: `8a5abc997a9d05264d4349a9d62fd0fc7d4ee630`  
Upstream issue: https://github.com/sgl-project/sglang/issues/38574  
Mirror issue: https://github.com/amdpilot-org/sglang/issues/781

## Verdict

Request changes. The candidate is useful partial hardening: it turns the generic
scheme-selection exception into an actionable error containing the MTP prefix,
matched target, format, and quantization details, and it adds a regression test.
It does **not** resolve the original dense-MTP startup failure. Construction with
`mtp.layers.0.self_attn.qkv_proj.weight`, a broad `Linear` W4A8 target, and no
MTP ignore entry still raises before checkpoint tensors are inspected.

The documentation also recommends `"mtp.*"` and `"model.mtp.*"` as possible
ignore entries. In the actual matcher, non-`re:` entries are exact names or
dotted suffixes, not globs. Both examples fail to ignore the tested MTP prefix.
`"re:mtp\\..*"` works only for an `mtp...` prefix, while
`"re:(model\\.)?mtp\\..*"` covers both tested prefix forms.

## Independent evidence

- Prepared base `358c163250ad3b1f62939b01ce1314a0a31a0365`: the real
  `CompressedTensorsConfig` + `ReplicatedLinear` construction reproduced the
  generic `NotImplementedError`; explicit `re:mtp\\..*` constructed a BF16
  `UnquantizedLinearMethod`. Raw: `/job/raw/j-587efa294695/base-repro.txt`.
- Exact candidate: the no-ignore case still raised, now with the full layer
  name, `Linear` target, `pack-quantized` format, W4 weights, A8 activations,
  and ignore guidance. Raw: `/job/raw/j-587efa294695/candidate-repro.txt`.
- Candidate regression: 3 passed. Raw:
  `/job/raw/j-587efa294695/candidate-regression.txt`.
- Candidate-relevant compressed-tensors tests: 21 passed. Raw:
  `/job/raw/j-587efa294695/candidate-relevant-tests.txt`.
- Adversarial ignore patterns and dense/packed pre-load inventories: the two
  plain `*` documentation examples raised; dense and packed inventories had
  the same pre-load exception because tensor names are unavailable at module
  construction. Raw: `/job/raw/j-587efa294695/candidate-adversarial.txt`.
- GPU: one AMD Instinct MI350X, `gfx950:sramecc+:xnack-`, Torch
  `2.11.0+rocm7.2`. The valid explicit-ignore BF16 linear path ran on GPU and
  differed from an independent CPU FP32 matmul by maximum absolute error
  `0.03134560585021973`, consistent with BF16 rounding.
- `git diff --check` passed. No native source changed, so no native rebuild was
  applicable. The imported compressed-tensors source was
  `/job/repo/python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py`.

## Classification and limitations

This is a partial fix plus test/documentation hardening, not a full
original-issue fix. Actual packed MTP checkpoint loading was not exercised;
the evidence only proves packed and dense inventories cannot affect the common
pre-load selection point. No Qwen3-Next model weights, 2x H100 NVL TP2/EP2
environment, or end-to-end NEXTN server startup was available. The available
GPU was AMD gfx950, so NVIDIA/H100 kernel behavior is unverified.

A broader compressed-tensors run produced 27 passes and six WNA16
backend-selection failures in the ROCm environment. Those tests are unrelated
to the candidate's changed path and are retained at
`/job/raw/j-587efa294695/candidate-focused-tests-valid.txt`; they are not used
as evidence for or against this candidate.
