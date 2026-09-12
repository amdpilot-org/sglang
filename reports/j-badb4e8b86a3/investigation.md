# GDN mixed conv-cache dtype correction

Upstream issue: https://github.com/sgl-project/sglang/issues/31719

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2370

Candidate PR: https://github.com/amdpilot-org/sglang/pull/2280

Independent review PR: https://github.com/amdpilot-org/sglang/pull/2335

The prepared base was `358c163250ad3b1f62939b01ce1314a0a31a0365`. I first
applied candidate commit `4b0d2276a4cf6ce3e6ed5d2c7f66dce81416d0ac` without
additional source changes. On the assigned AMD Instinct MI350X (gfx950), its
tracked-state cast succeeded, but the immediately following production ROCm
Triton causal convolution failed to compile for BF16 activations with either an
FP32 or FP16 cache. FP16 activations with a BF16 cache and same-dtype BF16 were
successful controls. The compiler diagnostics are retained in
`candidate_before.txt`.

The correction preserves the candidate's cache-boundary cast. For a prefill
whose configured cache dtype differs from the activation dtype, it gathers only
the active cache slots, converts that temporary to the activation dtype, runs
the existing causal-convolution kernel with identity indices, and casts the
updated slots back to the configured cache dtype. Same-dtype calls retain the
existing direct path.

`corrected_after.txt` records successful compilation and execution for both
review counterexamples and both controls. The fixture independently computes
the expected output and final cache with grouped `torch.nn.functional.conv1d`;
all values passed `assert_close`. The focused unit regression additionally
checks conversion, identity remapping, selected-slot writeback, and preservation
of unselected slots.

The reported Qwen3.6 weights, NVIDIA RTX 5090, CUDA 13.3, and SM120 were not
available. This work therefore qualifies the implicated GDN prefill boundary on
one gfx950/ROCm 7.2 GPU, not full-model serving, semantic accuracy, NVIDIA, or a
distributed workload. No native source changed, so a native rebuild was not
applicable.
