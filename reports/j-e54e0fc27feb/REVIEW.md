# Independent review: PR 769 at `3fb9e346d437e25b07aa699613d7a1eceb751410`

Recommendation: **accept**. The candidate fully resolves the original safety contract by rejecting unsupported non-128 W4A8 MoE group sizes before loading can proceed. It is a fail-closed fix, not support for group_size 64.

Upstream issue: https://github.com/sgl-project/sglang/issues/38573

Mirror issue: https://github.com/amdpilot-org/sglang/issues/795

## Evidence

On prepared base `358c163250ad3b1f62939b01ce1314a0a31a0365`, the real `CompressedTensorsW4AFP8MoE` constructor accepted the reported `group_size=64`. It also accepted independent boundaries 1, 127, 129, 256, and -64. Source inspection confirmed that weight-scale shapes use `self.group_size`, while the six calls in `cutlass_w4a8_moe.py` pass a literal chunk size of 128.

After temporarily checking out the exact candidate, the imported source path was `/job/repo/python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a8_fp8_moe.py`. The same independent probe accepted 128 and raised `ValueError` for every tested non-128 value. The candidate regression passed 4 tests, and the related DeepEP unit passed.

The candidate changes Python only. It does not change native/C++ code, so no native rebuild was needed. The installed source import was verified to come from the checkout rather than an unrelated wheel.

## Architecture limitations

The assigned device was one AMD Instinct MI350X, gfx950, using Torch 2.11.0+rocm7.2. The affected CUTLASS kernel is NVIDIA Hopper/sm90-specific; all 605 kernel numerical cases skipped on this host. No affected model weights were supplied. Consequently this review does not claim an H100 full-model serving reproduction, semantic-output comparison, direct kernel numerical validation, or validation of the issue's secondary K-per-rank TP/EP observation.

Those limitations do not leave a counterexample to the candidate's narrow contract: non-128 configurations can no longer reach the hardcoded-128 kernel through this scheme. Supporting group_size 64 remains separate future work.
