# Investigation: Qwen3.8 NVFP4 multimodal grounding

Upstream issue: https://github.com/sgl-project/sglang/issues/35949

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1267

## Result

The prepared base (`358c163250ad3b1f62939b01ce1314a0a31a0365`) already contains an
issue-specific implementation and regression coverage for the failure described
in the report. No additional source correction was justified.

`Qwen3_5Attention.forward_prepare_cuda_fused()` passes a three-axis MRoPE map
when positions are two-dimensional. The fused Triton kernel selects the temporal,
height, or width position independently for every rotary lane. The registered
kernel regression uses distinct values in all three position rows (so a mistaken
single-row implementation cannot pass) and compares fused Q/K output with
`MRotaryEmbedding.forward_native`. It covers both MRoPE layouts, including a
full-width rotary case, plus ordinary 1-D positions, gate copying, and rejected
position/map mismatches.

## Validation

The assigned device is an AMD Instinct MI350X (`gfx950`) under ROCm 7.2, not the
reporter's NVIDIA RTX 5090 / CUDA environment. The unmodified test invocation
failed during compilation because the prepared Triton stack emitted the
CUDA-only PDL instruction `griddepcontrol.launch_dependents`, which the AMD
assembler rejected. The full output is retained in
`raw/fused_qk_rmsnorm_rope_gate.log`.

To isolate the issue-specific numerical behavior without modifying the checkout,
the suite was rerun after setting the module's `_ENABLE_PDL` flag to `False` in
the test process. All three tests passed on the assigned GPU. The exact runner
and output are retained in `raw/fused_qk_rmsnorm_rope_gate_no_pdl.log`.

## Limitations

The reported `RadixArk/Qwen3.8-27B-NVFP4` weights, source image, and prompt were
not available. This investigation therefore does not reproduce or qualify the
model's grounding coordinates. The assigned AMD GPU also cannot verify the
CUDA/SM120 code generation or runtime behavior. The passing deterministic kernel
fixture establishes that the checked-out fused implementation agrees with the
native MRoPE reference on gfx950 once the unrelated CUDA-only PDL compiler path
is disabled; CUDA and end-to-end model confirmation remain required.
