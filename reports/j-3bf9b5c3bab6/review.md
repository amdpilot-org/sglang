# Independent review of amdpilot-org/sglang PR 1456

Candidate: `6893c04844bb113350c9a84936cf3bd48eccee49`

Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

Recommendation: **accept**

The recorded base reproduces the issue-specific control-flow failure: with an FP4 expert and dequantization requested, both ROCm FNUZ parameterizations enter the native AITER FP4 branch before calling `cast_e2m1fn_to_e4m3fn`. The candidate moves that conversion ahead of both the AITER early return and FNUZ normalization. Its focused regression fails twice on the base and passes at the exact candidate commit.

An independent GPU exercise used real packed tensors and real AITER post-load functions. With `SGLANG_USE_AITER=1`, the candidate converted the weights to `(1, 128, 128)` block FP8, cleared `is_fp4_expert`, and completed both FNUZ-disabled and FNUZ-enabled processing. The latter produced `torch.float8_e4m3fnuz` weights. This validates the repaired branch ordering and downstream block-FP8 preparation; it is not a full DeepSeek-V4 serving test.

No native source changes are present, so a native rebuild is not applicable. Imports resolved to `/job/repo/python/sglang/srt/layers/quantization/fp8.py`; AITER loaded its prepared native extension from `/tmp/amdpilot-repo-j-3bf9b5c3bab6/cache/aiter/module_aiter_core.so`.

Limitations: the assigned device is one AMD Instinct MI350X (`gfx950:sramecc+:xnack-`), not eight MI308X (`gfx942`) devices; DeepSeek-V4-Flash-0731 weights and TP8 were unavailable. Therefore full server startup, graph capture, dispatch across all ranks, GSM8K, and multi-node behavior were not reproduced. The candidate PR body also references mirror issue 1433 instead of the requested mirror issue 1493; that is a delivery-metadata defect, not a counterexample to the code fix.
