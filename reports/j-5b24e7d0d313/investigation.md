# Issue 32426 investigation

The affected checkpoint metadata identifies `Qwen3_5MoeForConditionalGeneration`
with compressed-tensors `nvfp4-pack-quantized` weights. SGLang v0.5.16's
FlashInfer CUTLASS migration left its fused W13 tensor in `[gate; up]` order,
although that backend consumes `[up; gate]`. This silently changes
`silu(gate) * up` and mismatches the associated block scales, explaining valid
but semantically garbled output.

The prepared main source already includes the issue-specific correction from
upstream PR #32430 (commit `3b5fdc954795f4fdb3addaeede3c910630fc3fa0`):

- `CompressedTensorsW4A4Nvfp4MoE.load_up_proj_weight_first` is true for
  FlashInfer CUTLASS and false for TRT-LLM, whose post-load path reorders W13.
- `CompressedTensorsFusedMoEMethod.create_weights` forwards that scheme choice
  to the common fused-MoE loader.
- The base compressed-tensors MoE scheme defaults to false, preserving other
  schemes' existing gate/up layout.

Upstream validated the correction on an NVIDIA RTX PRO 6000 Blackwell Server
Edition with `nm-testing/nvfp4_moe-e2e`: GSM8K changed from 0.00% accuracy and
100% truncation before to 93.93% accuracy and 0.76% truncation after. This
checkout adds the previously missing focused unit regression and records its
failing-before/passing-after results.

Local full-model validation is not claimed. The assigned accelerator is an AMD
Instinct MI350X (gfx950), while this compressed-tensors NVFP4 implementation is
explicitly NVIDIA Blackwell-only. The affected 21.9 GB weights were therefore
not downloaded, and the Ornith generation path remains locally unverified.

Upstream issue: https://github.com/sgl-project/sglang/issues/32426

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2091

Related fix: https://github.com/sgl-project/sglang/pull/32430
